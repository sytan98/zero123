import objaverse
from tqdm import tqdm
import multiprocessing

processes = multiprocessing.cpu_count()
print(objaverse.__version__)

uids = objaverse.load_uids()
print(len(uids), type(uids))


annotations = objaverse.load_annotations(uids[:10])
objects = objaverse.load_objects(uids=uids[:10], download_processes=processes)
