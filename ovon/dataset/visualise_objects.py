import argparse
import json
import os
import os.path as osp
import pickle
from typing import Dict, List, Union

import GPUtil
import habitat
import habitat_sim
import numpy as np
import torch
from habitat.config.default import get_agent_config, get_config
from habitat.config.default_structured_configs import HabitatSimSemanticSensorConfig
from habitat.config.read_write import read_write
from habitat.tasks.utils import compute_pixel_coverage
from habitat_sim._ext.habitat_sim_bindings import BBox, SemanticObject
from habitat_sim.agent.agent import AgentState
from habitat_sim.simulator import Simulator
from numpy import ndarray
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import get_hm3d_semantic_scenes
from PIL import Image
from torchvision.ops import masks_to_boxes
from torchvision.transforms import PILToTensor, ToPILImage
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm

SCENES_ROOT = "data/scene_datasets/hm3d"
NUM_GPUS = len(GPUtil.getAvailable(limit=256))
TASKS_PER_GPU = 12


def create_html(file_name, objects_mapping, visualised=True, threshold=0.05):
    html_head = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Objects Spatial Relationships</title>
    </head>"""
    html_style = """
    <style>
        /* Three image containers (use 25% for four, and 50% for two, etc) */
        .column {
        float: left;
        width: 20.00%;
        padding: 5px;
        }

        /* Clear floats after image containers */
        {
        box-sizing: border-box;
        }

        .row {
        display: flex;
        }
    </style>
    """
    cnt = 0
    html_body = ""
    for obj in objects_mapping.keys():
        # Visualized Objects
        if visualised and objects_mapping[obj][0][1] >= threshold:
            cnt += 1
            html_body += f"""<h3>{obj}</h3> 
                            <div class="row">
                            """
            for cov, frac, scene in objects_mapping[obj][:5]:
                html_body += f"""
                            <div class="column">
                                <img src="../images/objects/{scene}/{obj}.png" alt="{obj}" style="width:100%">
                                <h5>cov = {cov:.3f}, frac = {frac:.3f}</h5>
                            </div>
                            """
            html_body += "</div>"
        # Filtered Objects
        elif not visualised and objects_mapping[obj][0][1] < threshold:
            cnt += 1
            html_body += f"""<h3>{obj}</h3> 
                            <div class="row">
                            """
            for cov, frac, scene in objects_mapping[obj][:5]:
                html_body += f"""
                            <div class="column">
                                <img src="../images/objects/{scene}/{obj}.png" alt="{obj}" style="width:100%">
                                <h5>cov = {cov:.3f}, frac = {frac:.3f}</h5>
                            </div>
                            """
            html_body += "</div>"
    html_body = (
        f"""
                <body>
                <h2> Visualising {cnt} objects </h2>
                """
        + html_body
    )
    html_body += """</body>
                    </html>"""
    f = open(file_name, "w")
    f.write(html_head + html_style + html_body)
    f.close()


def is_on_ceiling(sim, aabb: BBox):
    point = np.asarray(aabb.center)
    snapped = sim.pathfinder.snap_point(point)

    # The snapped point is on the floor above
    # It is more than 20 cms above the object's upper edge
    if snapped[1] > point[1] + aabb.sizes[0] / 2 + 0.20:
        return True

    # Snapped point is on the ground
    if snapped[1] < point[1] - 1.5:
        return True
    return False


def _get_iou_pose(
    sim: Simulator, pose: AgentState, object1: SemanticObject, object2: SemanticObject
):
    agent = sim.get_agent(0)
    agent.set_state(pose)
    obs = sim.get_sensor_observations()
    cov1 = compute_pixel_coverage(obs["semantic"], object1.semantic_id)
    if object2 != None:
        cov2 = compute_pixel_coverage(obs["semantic"], object2.semantic_id)
        if cov1 > 0 and cov2 > 0:
            return cov1 + cov2, pose, "Success"
    else:
        return cov1, pose, "Success"
    return 0, pose, "Failure"


def get_best_viewpoint_with_posesampler(
    sim: Simulator,
    pose_sampler: PoseSampler,
    object1: SemanticObject,
    object2: Union[SemanticObject, None] = None,
):
    if isinstance(object2, SemanticObject):
        center = np.array((object1.aabb.center + object2.aabb.center) / 2)
        candidate_states = pose_sampler.sample_agent_poses_radially(
            search_center=center
        )
    else:
        candidate_states = pose_sampler.sample_agent_poses_radially(obj=object1)
    candidate_poses_ious = list(
        _get_iou_pose(sim, pos, object1, object2) for pos in candidate_states
    )
    candidate_poses_ious_filtered = [p for p in candidate_poses_ious if p[0] > 0]
    candidate_poses_sorted = sorted(
        candidate_poses_ious_filtered, key=lambda x: x[0], reverse=True
    )
    if candidate_poses_sorted:
        return True, candidate_poses_sorted[0]
    else:
        return False, None


def get_bounding_box(
    obs: List[Dict[str, ndarray]],
    object1: SemanticObject,
    object2: Union[SemanticObject, None] = None,
):
    H, W = obs["semantic"].shape[0], obs["semantic"].shape[1]
    masks = (obs["semantic"] == np.array([[(object1.semantic_id)]])).reshape((1, H, W))
    if object2:
        mask2 = (obs["semantic"] == np.array([[(object2.semantic_id)]])).reshape(
            (1, H, W)
        )
        masks = np.concatenate((masks, mask2), axis=0)
    try:
        boxes = masks_to_boxes(torch.from_numpy(masks))
        area = (boxes[0][2] - boxes[0][0]) * (boxes[0][3] - boxes[0][1])
        if object2:
            area += (boxes[1][2] - boxes[1][0]) * (boxes[1][3] - boxes[1][1])
        return (
            True,
            boxes,
            area / (H * W) * 1.00,
        )
    except ValueError:
        return False, None, None


def get_objects(sim, objects_info, scene_key, obj_mapping, pose_sampler):
    objects_visualized = []
    cnt = 0
    agent = sim.get_agent(0)

    filtered_objects = [
        obj for obj in objects_info if obj.category.name() in FILTERED_CATEGORIES
    ]
    if not osp.isdir(f"data/images/objects/{scene_key}"):
        os.mkdir(f"data/images/objects/{scene_key}")

    for object in filtered_objects:
        name = object.category.name().replace("/", " ")
        if is_on_ceiling(sim, object.aabb):
            continue

        check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, object)
        if check:
            cov, pose, _ = view
            agent.set_state(pose)
            obs = sim.get_sensor_observations()
            bb_check, bb, fraction = get_bounding_box(obs, object)
            if (
                bb_check
                and fraction > 0
                and (
                    name not in obj_mapping
                    or scene_key not in obj_mapping[name]
                    or cov > obj_mapping[name][scene_key][0]
                )
            ):
                # Add object visualization to mapping
                if name not in obj_mapping.keys():
                    obj_mapping[name] = {}
                obj_mapping[name][scene_key] = (cov, fraction)

                # Draw bounding box and save image
                img = Image.fromarray(obs["rgb"][:, :, :3], "RGB")
                drawn_boxes = draw_bounding_boxes(
                    PILToTensor()(img), bb, colors="red", width=3
                )
                (ToPILImage()(drawn_boxes)).convert("RGB").save(
                    f"data/images/objects/{scene_key}/{name}.png"
                )
                objects_visualized.append(object.category.name().strip())
                cnt += 1

    print(
        f"Visualised {cnt} number of objects for scene {scene_key} out of total {len(filtered_objects)}!"
    )


def get_objnav_config(i, scene):
    CFG = "habitat-lab/habitat-lab/habitat/config/benchmark/nav/objectnav/objectnav_hm3d.yaml"
    SCENE_CFG = f"{SCENES_ROOT}/hm3d_annotated_basis.scene_dataset_config.json"
    objnav_config = get_config(CFG)

    with read_write(objnav_config):
        agent_config = get_agent_config(objnav_config.habitat.simulator)

        # Stretch agent
        agent_config.height = 1.41
        agent_config.radius = 0.17

        sensor_pos = [0, 1.31, 0]

        del agent_config.sim_sensors["depth_sensor"]
        agent_config.sim_sensors.update(
            {"semantic_sensor": HabitatSimSemanticSensorConfig()}
        )
        FOV = 90

        for sensor, sensor_config in agent_config.sim_sensors.items():
            agent_config.sim_sensors[sensor].hfov = FOV
            agent_config.sim_sensors[sensor].width //= 2
            agent_config.sim_sensors[sensor].height //= 2
            agent_config.sim_sensors[sensor].position = sensor_pos

        objnav_config.habitat.task.measurements = {}

        deviceIds = GPUtil.getAvailable(
            order="memory", limit=1, maxLoad=1.0, maxMemory=1.0
        )
        if i < NUM_GPUS * TASKS_PER_GPU or len(deviceIds) == 0:
            deviceId = i % NUM_GPUS
        else:
            deviceId = deviceIds[0]
        objnav_config.habitat.simulator.habitat_sim_v0.gpu_device_id = (
            deviceId  # i % NUM_GPUS
        )
        objnav_config.habitat.dataset.scenes_dir = "./data/scene_datasets/"
        objnav_config.habitat.dataset.split = "train"
        objnav_config.habitat.simulator.scene = scene
        objnav_config.habitat.simulator.scene_dataset = SCENE_CFG
    return objnav_config


def get_simulator(objnav_config):
    sim = habitat.sims.make_sim("Sim-v0", config=objnav_config.habitat.simulator)
    navmesh_settings = habitat_sim.NavMeshSettings()
    navmesh_settings.set_defaults()
    navmesh_settings.agent_radius = (
        objnav_config.habitat.simulator.agents.main_agent.radius
    )
    navmesh_settings.agent_height = (
        objnav_config.habitat.simulator.agents.main_agent.height
    )
    sim.recompute_navmesh(sim.pathfinder, navmesh_settings)
    return sim


def main():
    object_map = {}  # map between object instance -> (coverage, scene)
    # sort for each object and generate 4-5 visualisation
    HM3D_SCENES = get_hm3d_semantic_scenes("data/scene_datasets/hm3d")
    for i, scene in tqdm(enumerate(list(HM3D_SCENES[split]))):
        scene_key = os.path.basename(scene).split(".")[0]
        cfg = get_objnav_config(i, scene_key)
        sim = get_simulator(cfg)
        objects_info = sim.semantic_scene.objects
        pose_sampler = PoseSampler(
            sim=sim,
            r_min=0.5,
            r_max=2.0,
            r_step=0.5,
            rot_deg_delta=10.0,
            h_min=0.8,
            h_max=1.4,
            sample_lookat_deg_delta=5.0,
        )
        print(f"Starting scene: {scene_key}")
        get_objects(sim, objects_info, scene_key, object_map, pose_sampler)
        sim.close()

    sorted_object_mapping = {}
    print(f"Total number of objects visualized is {len(object_map.keys())}")
    for obj in object_map.keys():
        sorted_object_mapping[obj] = []
        for scene, (cov, fraction) in object_map[obj].items():
            sorted_object_mapping[obj].append((cov, fraction, scene))
        sorted_object_mapping[obj] = sorted(sorted_object_mapping[obj], reverse=True)
    with open(f"data/object_images_{split}.pickle", "wb") as handle:
        pickle.dump(sorted_object_mapping, handle, protocol=pickle.HIGHEST_PROTOCOL)

    for threshold in [0.025, 0.05, 0.1, 0.15, 0.20]:
        for vis in [True, False]:
            if vis:
                create_html(
                    f"data/webpage/objects_visualized_{split}_{threshold}.html",
                    sorted_object_mapping,
                    visualised=vis,
                    threshold=threshold,
                )
            else:
                create_html(
                    f"data/webpage/objects_filtered_{split}_{threshold}.html",
                    sorted_object_mapping,
                    visualised=vis,
                    threshold=threshold,
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-s",
        "--split",
        help="split of data to be used",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-f",
        "--filtered_data",
        help="path of json which contains the filtered categories",
        type=str,
        required=True,
    )
    args = parser.parse_args()
    split = args.split
    f = open(args.filtered_data)
    FILTERED_CATEGORIES = json.load(f)
    main()
