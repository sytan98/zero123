import glob
import json
import multiprocessing
import shutil
import subprocess
import time
from dataclasses import dataclass
from typing import Optional
import os

import boto3
import tyro
import wandb
import math
import signal

blender_path = " ~/workspace/computer_vision/home/hj453/code/blender_download/Blender_rendering/software/blender-3.2.2-linux-x64/blender"


@dataclass
class Args:
    workers_per_gpu: int
    """number of workers per gpu"""

    input_models_path: str
    """Path to a json file containing a list of 3D object files"""

    upload_to_s3: bool = False
    """Whether to upload the rendered images to S3"""

    log_to_wandb: bool = False
    """Whether to log the progress to wandb"""

    num_gpus: int = -1
    """number of gpus to use. -1 means all available gpus"""


def worker(
    queue: multiprocessing.JoinableQueue,
    count: multiprocessing.Value,
    gpu: int,
    s3: Optional[boto3.client],
) -> None:
    while True:
        item = queue.get()
        if item is None:
            break

        view_path = os.path.join(".objaverse/hf-objaverse-v1/views_whole_sphere", item.split("/")[-1][:-4])
        if os.path.exists(view_path):
            queue.task_done()
            print("========", item, "rendered", "========")
            continue
        # else:
        #     os.makedirs(view_path, exist_ok = True)

        # Perform some operation on the item
        print(item, gpu)
        command = (
            # f"export DISPLAY=:0.{gpu} &&"
            # f" GOMP_CPU_AFFINITY='0-47' OMP_NUM_THREADS=48 OMP_SCHEDULE=STATIC OMP_PROC_BIND=CLOSE "
            # f" CUDA_VISIBLE_DEVICES={gpu} "
            f" {blender_path} -b -P scripts/blender_script.py --python-use-system-env -- --object_path /home/sytan98/{item} --num_images 1"
        )
        print(command)
        subprocess.run(command, shell=True)

        with count.get_lock():
            count.value += 1

        queue.task_done()


if __name__ == "__main__":
    args = tyro.cli(Args)

    s3 = boto3.client("s3") if args.upload_to_s3 else None
    queue = multiprocessing.JoinableQueue()
    count = multiprocessing.Value("i", 0)

    if args.log_to_wandb:
        wandb.init(project="objaverse-rendering", entity="prior-ai2")
    processes = []
    # Start worker processes on each of the GPUs
    for gpu_i in range(args.num_gpus):
        for worker_i in range(args.workers_per_gpu):
            worker_i = gpu_i * args.workers_per_gpu + worker_i
            process = multiprocessing.Process(target=worker, args=(queue, count, gpu_i, s3))
            process.daemon = True
            process.start()
            processes.append(process)

    try:
        # Add items to the queue
        with open(args.input_models_path, "r") as f:
            model_paths = json.load(f)
        # import ipdb; ipdb.set_trace()
        model_keys = list(model_paths.keys())

        model_path_length = 30
        model_keys = list(model_paths.keys())[:model_path_length]

        for item in model_keys:
            queue.put(os.path.join(".objaverse/hf-objaverse-v1", model_paths[item]))

        # update the wandb count
        if args.log_to_wandb:
            while True:
                time.sleep(5)
                wandb.log(
                    {
                        "count": count.value,
                        "total": model_path_length,
                        "progress": count.value / model_path_length,
                    }
                )
                if count.value == model_path_length:
                    break

        # Wait for all tasks to be completed
        queue.join()

        # Add sentinels to the queue to stop the worker processes
        for i in range(args.num_gpus * args.workers_per_gpu):
            queue.put(None)
    except KeyboardInterrupt:
        print("Received keyboard interrupt. Terminating processes.")
        for p in processes:
            os.kill(p.pid, signal.SIGINT)
