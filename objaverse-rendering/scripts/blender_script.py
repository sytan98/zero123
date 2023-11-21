"""Blender script to render images of 3D models.

This script is used to render images of 3D models. It takes in a list of paths
to .glb files and renders images of each model. The images are from rotating the
object around the origin. The images are saved to the output directory.

Example usage:
    blender -b -P blender_script.py -- \
        --object_path my_object.glb \
        --output_dir ./views \
        --engine CYCLES \
        --scale 0.8 \
        --num_images 12 \
        --camera_dist 1.2

Here, input_model_paths.json is a json file containing a list of paths to .glb.
"""

import argparse
import glob
import json
import math
import os
from os.path import basename
import random
import sys
import time
import urllib.request
import uuid
from typing import Tuple
from mathutils import Vector, Matrix
import numpy as np

import bpy
from mathutils import Vector

parser = argparse.ArgumentParser()
parser.add_argument(
    "--object_path",
    type=str,
    required=True,
    help="Path to the object file",
)
parser.add_argument("--output_dir", type=str, default=".objaverse/hf-objaverse-v1/views_whole_sphere")
parser.add_argument("--engine", type=str, default="CYCLES", choices=["CYCLES", "BLENDER_EEVEE"])
parser.add_argument("--scale", type=float, default=0.8)
parser.add_argument("--num_images", type=int, default=8)
parser.add_argument("--camera_dist", type=int, default=1.2)
parser.add_argument(
    "--test_light_dir",
    type=str,
    default="/home/sytan98/workspace/zero123/objaverse-rendering/light_probes/high_res_envmaps_2k",
    help="directory containing the test (novel) light probes",
)

argv = sys.argv[sys.argv.index("--") + 1 :]
args = parser.parse_args(argv)

print("===================", args.engine, "===================")

context = bpy.context
scene = context.scene
render = scene.render

cam = scene.objects["Camera"]
cam.location = (0, 1.2, 0)
cam.data.lens = 35
cam.data.sensor_width = 32

cam_constraint = cam.constraints.new(type="TRACK_TO")
cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
cam_constraint.up_axis = "UP_Y"

# setup lighting
bpy.ops.object.light_add(type="AREA")
light2 = bpy.data.lights["Area"]
light2.energy = 3000
bpy.data.objects["Area"].location[2] = 0.5
bpy.data.objects["Area"].scale[0] = 100
bpy.data.objects["Area"].scale[1] = 100
bpy.data.objects["Area"].scale[2] = 100

render.engine = args.engine
render.image_settings.file_format = "PNG"
render.image_settings.color_mode = "RGBA"
render.resolution_x = 512
render.resolution_y = 512
render.resolution_percentage = 100

scene.cycles.device = "GPU"
scene.cycles.samples = 128
scene.cycles.diffuse_bounces = 1
scene.cycles.glossy_bounces = 1
scene.cycles.transparent_max_bounces = 3
scene.cycles.transmission_bounces = 3
scene.cycles.filter_width = 0.01
scene.cycles.use_denoising = True
scene.render.film_transparent = True

bpy.context.preferences.addons["cycles"].preferences.get_devices()
# Set the device_type
bpy.context.preferences.addons["cycles"].preferences.compute_device_type = "CUDA"  # or "OPENCL"


def sample_point_on_sphere(radius: float) -> Tuple[float, float, float]:
    theta = random.random() * 2 * math.pi
    phi = math.acos(2 * random.random() - 1)
    return (
        radius * math.sin(phi) * math.cos(theta),
        radius * math.sin(phi) * math.sin(theta),
        radius * math.cos(phi),
    )


def sample_spherical(radius=3.0, maxz=3.0, minz=0.0):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        vec[2] = np.abs(vec[2])
        vec = vec / np.linalg.norm(vec, axis=0) * radius
        if maxz > vec[2] > minz:
            correct = True
    return vec


def sample_spherical(radius_min=1.5, radius_max=2.0, maxz=1.6, minz=-0.75):
    correct = False
    while not correct:
        vec = np.random.uniform(-1, 1, 3)
        #         vec[2] = np.abs(vec[2])
        radius = np.random.uniform(radius_min, radius_max, 1)
        vec = vec / np.linalg.norm(vec, axis=0) * radius[0]
        if maxz > vec[2] > minz:
            correct = True
    return vec


def randomize_camera():
    elevation = random.uniform(0.0, 90.0)
    azimuth = random.uniform(0.0, 360)
    distance = random.uniform(0.8, 1.6)
    return set_camera_location(elevation, azimuth, distance)


def set_camera_location(elevation, azimuth, distance):
    # from https://blender.stackexchange.com/questions/18530/
    x, y, z = sample_spherical(radius_min=1.5, radius_max=2.2, maxz=2.2, minz=-2.2)
    camera = bpy.data.objects["Camera"]
    camera.location = x, y, z

    direction = -camera.location
    rot_quat = direction.to_track_quat("-Z", "Y")
    camera.rotation_euler = rot_quat.to_euler()
    return camera


def randomize_lighting() -> None:
    light2.energy = random.uniform(300, 600)
    bpy.data.objects["Area"].location[0] = random.uniform(-1.0, 1.0)
    bpy.data.objects["Area"].location[1] = random.uniform(-1.0, 1.0)
    bpy.data.objects["Area"].location[2] = random.uniform(0.5, 1.5)


def add_light_env(env=(1, 1, 1, 1), strength=1, rot_vec_rad=(0, 0, 0), scale=(1, 1, 1)):
    r"""Adds environment lighting.

    Args:
        env (tuple(float) or str, optional): Environment map. If tuple,
            it's RGB or RGBA, each element of which :math:`\in [0,1]`.
            Otherwise, it's the path to an image.
        strength (float, optional): Light intensity.
        rot_vec_rad (tuple(float), optional): Rotations in radians around x,
            y and z.
        scale (tuple(float), optional): If all changed simultaneously,
            then no effects.
    """

    engine = bpy.context.scene.render.engine
    assert engine == "CYCLES", "Rendering engine is not Cycles"

    if isinstance(env, str):
        bpy.data.images.load(env, check_existing=True)
        env = bpy.data.images[basename(env)]
    else:
        if len(env) == 3:
            env += (1,)
        assert len(env) == 4, "If tuple, env must be of length 3 or 4"

    world = bpy.context.scene.world
    world.use_nodes = True
    node_tree = world.node_tree
    nodes = node_tree.nodes
    links = node_tree.links

    bg_node = nodes.new("ShaderNodeBackground")
    links.new(bg_node.outputs["Background"], nodes["World Output"].inputs["Surface"])

    if isinstance(env, tuple):
        # Color
        bg_node.inputs["Color"].default_value = env
        print(("Environment is pure color, " "so rotation and scale have no effect"))
    else:
        # Environment map
        texcoord_node = nodes.new("ShaderNodeTexCoord")
        env_node = nodes.new("ShaderNodeTexEnvironment")
        env_node.image = env
        mapping_node = nodes.new("ShaderNodeMapping")
        mapping_node.inputs["Rotation"].default_value = rot_vec_rad
        mapping_node.inputs["Scale"].default_value = scale
        links.new(texcoord_node.outputs["Generated"], mapping_node.inputs["Vector"])
        links.new(mapping_node.outputs["Vector"], env_node.inputs["Vector"])
        links.new(env_node.outputs["Color"], bg_node.inputs["Color"])

    bg_node.inputs["Strength"].default_value = strength
    print("Environment light added")


def reset_lighting() -> None:
    light2.energy = 1000
    bpy.data.objects["Area"].location[0] = 0
    bpy.data.objects["Area"].location[1] = 0
    bpy.data.objects["Area"].location[2] = 0.5


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


# load the glb model
def load_object(object_path: str) -> None:
    """Loads a glb model into the scene."""
    if object_path.endswith(".glb"):
        bpy.ops.import_scene.gltf(filepath=object_path, merge_vertices=True)
    elif object_path.endswith(".fbx"):
        bpy.ops.import_scene.fbx(filepath=object_path)
    else:
        raise ValueError(f"Unsupported file type: {object_path}")


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


# function from https://github.com/panmari/stanford-shapenet-renderer/blob/master/render_blender.py
def get_3x4_RT_matrix_from_blender(cam):
    # bcam stands for blender camera
    # R_bcam2cv = Matrix(
    #     ((1, 0,  0),
    #     (0, 1, 0),
    #     (0, 0, 1)))

    # Transpose since the rotation is object rotation,
    # and we want coordinate rotation
    # R_world2bcam = cam.rotation_euler.to_matrix().transposed()
    # T_world2bcam = -1*R_world2bcam @ location
    #
    # Use matrix_world instead to account for all constraints
    location, rotation = cam.matrix_world.decompose()[0:2]
    R_world2bcam = rotation.to_matrix().transposed()

    # Convert camera location to translation vector used in coordinate changes
    # T_world2bcam = -1*R_world2bcam @ cam.location
    # Use location from matrix_world to account for constraints:
    T_world2bcam = -1 * R_world2bcam @ location

    # # Build the coordinate transform matrix from world to computer vision camera
    # R_world2cv = R_bcam2cv@R_world2bcam
    # T_world2cv = R_bcam2cv@T_world2bcam

    # put into 3x4 matrix
    RT = Matrix(
        (
            R_world2bcam[0][:] + (T_world2bcam[0],),
            R_world2bcam[1][:] + (T_world2bcam[1],),
            R_world2bcam[2][:] + (T_world2bcam[2],),
        )
    )
    return RT


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    scale = 1 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def has_materials() -> bool:
    """Check if any object in the scene has materials."""
    # material_set = set()
    # for object in context.scene.objects:
    #     for material in object.material_slots:
    #         material_set.add(material)

    # for material in material_set:
    #     print(material.name)
    for obj in bpy.context.scene.objects:
        if obj.type == "MESH" and obj.data.materials:
            return True
    return False


def save_images(object_file: str) -> None:
    """Saves rendered images of the object in the scene."""
    os.makedirs(args.output_dir, exist_ok=True)

    reset_scene()

    # load the object
    load_object(object_file)
    object_uid = os.path.basename(object_file).split(".")[0]
    normalize_scene()

    # Ignore object if object has no material
    if not has_materials():
        return

    # create an empty object to track
    empty = bpy.data.objects.new("Empty", None)
    scene.collection.objects.link(empty)
    cam_constraint.target = empty

    bpy.context.scene.render.film_transparent = False
    if args.test_light_dir is None:
        randomize_lighting()
        for i in range(args.num_images):
            # # set the camera position
            # theta = (i / args.num_images) * math.pi * 2
            # phi = math.radians(60)
            # point = (
            #     args.camera_dist * math.sin(phi) * math.cos(theta),
            #     args.camera_dist * math.sin(phi) * math.sin(theta),
            #     args.camera_dist * math.cos(phi),
            # )
            # # reset_lighting()
            # cam.location = point

            # set camera
            camera = randomize_camera()

            # render the image
            render_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.png")
            scene.render.filepath = render_path
            bpy.ops.render.render(write_still=True)

            # save camera RT matrix
            RT = get_3x4_RT_matrix_from_blender(camera)
            RT_path = os.path.join(args.output_dir, object_uid, f"{i:03d}.npy")
            np.save(RT_path, RT)
    else:
        # Render relit ground truth
        for envmap_name in glob.glob(os.path.join(args.test_light_dir, "*.hdr"), recursive=True):
            envmap_path = envmap_name

            add_light_env(env=envmap_path, strength=1.0)
            for i in range(args.num_images):
                # set camera
                camera = randomize_camera()

                # render the image
                render_path = os.path.join(
                    args.output_dir, object_uid, f"{i:03d}_{basename(envmap_name).split('.')[0]}.png"
                )
                scene.render.filepath = render_path
                bpy.ops.render.render(write_still=True)

                # save camera RT matrix
                RT = get_3x4_RT_matrix_from_blender(camera)
                RT_path = os.path.join(
                    args.output_dir, object_uid, f"{i:03d}_{basename(envmap_name).split('.')[0]}.npy"
                )
                np.save(RT_path, RT)


def download_object(object_url: str) -> str:
    """Download the object and return the path."""
    # uid = uuid.uuid4()
    uid = object_url.split("/")[-1].split(".")[0]
    tmp_local_path = os.path.join("tmp-objects", f"{uid}.glb" + ".tmp")
    local_path = os.path.join("tmp-objects", f"{uid}.glb")
    # wget the file and put it in local_path
    os.makedirs(os.path.dirname(tmp_local_path), exist_ok=True)
    urllib.request.urlretrieve(object_url, tmp_local_path)
    os.rename(tmp_local_path, local_path)
    # get the absolute path
    local_path = os.path.abspath(local_path)
    return local_path


if __name__ == "__main__":
    try:
        start_i = time.time()
        if args.object_path.startswith("http"):
            local_path = download_object(args.object_path)
        else:
            local_path = args.object_path
        save_images(local_path)
        end_i = time.time()
        print("Finished", local_path, "in", end_i - start_i, "seconds")
        # delete the object if it was downloaded
        if args.object_path.startswith("http"):
            os.remove(local_path)
    except Exception as e:
        print("Failed to render", args.object_path)
        print(e)
