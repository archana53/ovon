import argparse
import itertools
import json
import os
import os.path as osp
import pickle

import matplotlib.pyplot as plt
import numpy as np
from habitat.tasks.utils import compute_pixel_coverage
from habitat.utils.geometry_utils import quaternion_from_two_vectors
from ovon.dataset.pose_sampler import PoseSampler
from ovon.dataset.semantic_utils import get_hm3d_semantic_scenes
from ovon.dataset.visualise_objects import (
    get_best_viewpoint_with_posesampler,
    get_bounding_box,
    get_objnav_config,
    get_simulator,
)
from PIL import Image, ImageDraw
from torchvision.ops import masks_to_boxes
from torchvision.transforms import PILToTensor, ToPILImage
from torchvision.utils import draw_bounding_boxes
from tqdm import tqdm


def create_html(relationships, file_name):
    html_text = """
    <html>
    <head>
        <meta charset="utf-8">
        <title>Objects Spatial Relationships</title>
    </head>
    <style>
        .image-list-small {
        font-family: Arial, Helvetica, sans-serif;
        margin: 0 auto;
        text-align: center;
        max-width: 640px;
        padding: 0;
        }

        .image-list-small li {
        display: inline-block;
        width: 250px;
        margin: 0 12px 30px;
        }


        /* Photo */

        .image-list-small li > a {
        display: block;
        text-decoration: none;
        background-size: cover;
        background-repeat: no-repeat;
        height: 250px;
        width: 250 px;
        margin: 0;
        padding: 0;
        border: 4px solid #ffffff;
        outline: 1px solid #d0d0d0;
        box-shadow: 0 2px 1px #DDD;
        }

        .image-list-small .details {
        margin-top: 13px;
        }


        /* Title */

        .image-list-small .details h3 {
        display: block;
        font-size: 12px;
        margin: 0 0 3px 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        }

        .image-list-small .details h3 a {
        color: #303030;
        text-decoration: none;
        }

        .image-list-small .details .image-author {
        display: block;
        color: #717171;
        font-size: 11px;
        font-weight: normal;
        margin: 0;
        }
    </style>
    <body>
    <ul class="image-list-small">"""
    for rel in relationships:
        html_text += f"""
            <li>
            <a href="#" style="background-image: url('../images/relationships/relationship_{rel}.png');"></a>
            <div class="details">
                <h3><a href="#"> {rel} </a></h3>
            </div>
            </li>
            """
    html_text += """</ul>
                    </body>
                    </html>"""
    f = open(file_name, "w")
    f.write(html_text)
    f.close()


"""

def set_agent_state(pt, q):
    agent_state = agent.get_state()
    agent_state.position = pt  # world space
    agent_state.rotation = q
    agent.set_state(agent_state)
    obs = sim.get_sensor_observations()
    return obs


def get_best_viewpoint(sim, pt, id1, id2):
    "Returns the best viewpoint of objects of id1 and id2"
    pf = sim.pathfinder
    x_coords = np.linspace(pt[0] - 1.5, pt[0] + 1.5, 20)
    z_coords = np.linspace(pt[2] - 1.5, pt[2] + 1.5, 20)
    y = pt[1]
    maxcov = 0
    for x, z in list(itertools.product(x_coords, z_coords)):
        cov = 0
        pt_new = pf.snap_point([x, y, z])
        goal_direction = pt - pt_new
        goal_direction[1] = 0
        q = _direction_to_quaternion(goal_direction)
        obs = set_agent_state(pt_new, q)
        cov += compute_pixel_coverage(obs["semantic"], id1)
        cov += compute_pixel_coverage(obs["semantic"], id2)
        if cov > maxcov:
            best_obs = obs
            maxcov = cov
    return maxcov, best_obs


def get_bounding_box(obs, obj_a, obj_b):
    a_args = np.argwhere(obs["semantic"] == (obj_a.semantic_id))
    b_args = np.argwhere(obs["semantic"] == (obj_b.semantic_id))
    if a_args.shape[0] < 150 or b_args.shape[0] < 150:
        print(
            obj_a.category.name(),
            a_args.shape[0],
            obj_b.category.name(),
            b_args.shape[0],
        )
        return False, None, None
    try:
        bb_a = (
            np.min(a_args[:, 0]),
            np.max(a_args[:, 0]),
            np.min(a_args[:, 1]),
            np.max(a_args[:, 1]),
        )
        try:
            bb_b = (
                np.min(b_args[:, 0]),
                np.max(b_args[:, 0]),
                np.min(b_args[:, 1]),
                np.max(b_args[:, 1]),
            )
            return True, bb_a, bb_b
        except ValueError:
            return False, None, None
    except ValueError:
        return False, None, None


def confirm_orientation(sim, obj_a, obj_b, reln_prompt, direction):
    pf = sim.pathfinder
    center = (obj_a.aabb.center + obj_b.aabb.center) / 2
    pt = pf.snap_point()


def visualize_relationship(sim, obj_a, obj_b, reln_prompt):
    # Find location best location to view both obj_a and obj_b
    objs_center = (obj_a.aabb.center + obj_b.aabb.center) / 2
    cov, obs = get_best_viewpoint(
        sim, objs_center, obj_a.semantic_id, obj_b.semantic_id
    )
    # Donot visualize if objects are too small to notice
    i = Image.fromarray(obs["color"][:, :, :3], "RGB")
    draw = ImageDraw.Draw(i)
    check, bb_a, bb_b = get_bounding_box(obs, obj_a, obj_b)
    if check:
        draw.rectangle([(bb_a[3], bb_a[1]), (bb_a[2], bb_a[0])], outline="red", width=5)
        draw.rectangle([(bb_b[3], bb_b[1]), (bb_b[2], bb_b[0])], outline="red", width=5)
        i.save(f"data/images/relationships/relationship_{reln_prompt}.png")
    else:
        print(reln_prompt)
    return check


def a_leftto_b(self, a, b, dx=0.05, delta=0.1):
    # Objects are close by in x/z axis distance
    distance_x = (b.aabb.center[0] - b.aabb.sizes[0] / 2) - (
        a.aabb.center[0] + a.aabb.sizes[0] / 2
    )
    # distance_z = (b.aabb.center[2] - b.aabb.sizes[2]/2) - (a.aabb.center[2] + a.aabb.sizes[2]/2)
    total_dist = np.linalg.norm(a.aabb.center - b.aabb.center)
    if (distance_x <= dx and distance_x >= 0) and total_dist <= delta:
        return (True, f"{a.category.name()} left of {b.category.name()}")

    return (False, "")


def a_rightto_b(self, a, b, dx=0.05, delta=0.1):
    # Objects are close by in x/z axis distance
    distance_x = (a.aabb.center[0] - a.aabb.sizes[0] / 2) - (
        b.aabb.center[0] + b.aabb.sizes[0] / 2
    )
    # distance_z = (a.aabb.center[2] - a.aabb.sizes[2]/2) - (b.aabb.center[2] + b.aabb.sizes[2]/2)
    total_dist = np.linalg.norm(a.aabb.center - b.aabb.center)
    if (distance_x <= dx and distance_x >= 0) and total_dist <= delta:
        return (True, f"{a.category.name()} right of {b.category.name()}")

    return (False, "")


def a_above_b(self, a, b, dy=0.05, delta=0.1):
    # a should be bigger than b
    distance_y = (a.aabb.center[1] - a.aabb.sizes[1] / 2) - (
        b.aabb.center[1] + b.aabb.sizes[1] / 2
    )
    total_dist = np.linalg.norm(a.aabb.center - b.aabb.center)
    if distance_y <= dy and distance_y >= 0 and total_dist <= delta:
        return (True, f"{a.category.name()} above {b.category.name()}")

    return (False, "")


def a_below_b(self, a, b, dy=0.1, delta=0.2):
    # Center of a should lie in x and z axis bounds of b
    # Center of a_y should be lesser then b_y by a margin
    # a should be bigger than b

    distance_y = (b.aabb.center[1] - b.aabb.sizes[1] / 2) - (
        a.aabb.center[1] + a.aabb.sizes[1] / 2
    )
    total_dist = np.linalg.norm(a.aabb.center - b.aabb.center)
    if (distance_y <= dy and distance_y) >= 0 and total_dist <= delta:
        return (True, f"{a.category.name()} below {b.category.name()}")

    return (False, "")

"""


def a_near_b(a, b, delta=0.5):
    if np.linalg.norm(a.aabb.center - b.aabb.center) <= delta:
        return (True, f"{a.category.name()} near {b.category.name()}")
    else:
        return (False, "")


"""

def a_inside_b(a, b):
    a1 = a.aabb.center - a.aabb.sizes / 2
    a2 = a.aabb.center + a.aabb.sizes / 2

    b1 = b.aabb.center - b.aabb.sizes / 2
    b2 = b.aabb.center + b.aabb.sizes / 2
    check = True
    for i in range(len(a.aabb.center)):
        if a1[i] < b1[i] or a2[i] > b2[i]:
            check = False
    if check:
        return (True, f"{a.category.name()} inside {b.category.name()}")
    else:
        return (False, "")



def get_relationships_from3d(sim, objects, d=0.1, delta=0.5):
    f = open("data/obj/filtered_raw_categories.json")
    relations = ["above", "below", "leftto", "rightto"]
    FILTERED_CATEGORIES = json.load(f)
    filtered_objects = [
        obj for obj in objects if obj.category.name() in FILTERED_CATEGORIES
    ]
    mapping = {}
    obj_relationships = []
    for a in filtered_objects:
        mapping[a.category.name()] = {
            "leftto": [],
            "rightto": [],
            "above": [],
            "below": [],
        }
        for b in filtered_objects:
            if a.category.name() != b.category.name():
                for r in relations:
                    func = getattr(functions, "a_" + r + "_b")
                    val, prompt = func(a, b, d, delta)
                    if val == True and not (
                        np.array_equal(a.aabb.center, b.aabb.center)
                    ):
                        mapping[a.category.name()][r].append(b.category.name())
                        if visualize_relationship(sim, a, b, prompt):
                            obj_relationships.append(prompt)
                            mapping[a.category.name()][r].append(b.category.name())

    create_html(obj_relationships, "data/webpage/relationships_3d.html")

"""


def get_relation_2d(sim, pose_sampler, a, b, mapping):
    agent = sim.get_agent(0)
    check, view = get_best_viewpoint_with_posesampler(sim, pose_sampler, a, b)
    name_b = b.category.name()
    name_a = a.category.name()
    if check:
        cov, pose, _ = view
        agent.set_state(pose)
        obs = sim.get_sensor_observations()
        bb_check, bb, fraction = get_bounding_box(obs, a, b)
        if bb_check and fraction > 0:
            center1 = np.array([(bb[0][0] + bb[0][2]) / 2, (bb[0][1] + bb[0][3]) / 2])
            center2 = np.array([(bb[1][0] + bb[1][2]) / 2, (bb[1][1] + bb[1][3]) / 2])
            x_disp, y_disp = center2 - center1
            if abs(x_disp) > abs(y_disp):  # left/right/near relationship
                if x_disp > 0:
                    rel = f"{name_b} near (right) {name_a}"
                else:
                    rel = f"{name_b} near (left) {name_a}"
            else:  # above/below relationship
                if y_disp > 0:
                    rel = f"{name_b} below {name_a}"
                else:
                    rel = f"{name_b} above {name_a}"
            img = Image.fromarray(obs["rgb"][:, :, :3], "RGB")
            drawn_boxes = draw_bounding_boxes(
                PILToTensor()(img), bb, colors="red", width=3
            )
            return True, rel, drawn_boxes
    return False, None, None


def get_relationships_from2d(sim, objects, scene_key, pose_sampler, mapping):
    filtered_objects = [
        obj for obj in objects if obj.category.name() in FILTERED_CATEGORIES
    ]
    if not osp.isdir(f"data/images/relationships_2d/{scene_key}"):
        os.mkdir(f"data/images/relationships_2d/{scene_key}")
    obj_relationships = []
    for a in filtered_objects:
        for b in filtered_objects:
            name_a = a.category.name()
            name_b = b.category.name()
            volume_a = np.prod((a.aabb.sizes))
            volume_b = np.prod((b.aabb.sizes))
            if name_a != name_b and volume_a > volume_b and a_near_b(a, b)[0]:
                print(
                    f"Finding relation between {a.category.name()}  and {b.category.name()}"
                )
                check, rel, img = get_relation_2d(sim, pose_sampler, a, b, mapping)
                if check:
                    print(rel)
                    (ToPILImage()(img)).convert("RGB").save(
                        f"data/images/relationships_2d/{scene_key}/{rel}.png"
                    )

    # create_html(obj_relationships, "data/webpage/relationships_2d.html")


def main():
    relationship_map = (
        {}
    )  # map between scene -> relationship -> obja -> objb -> coverage
    # sort for each object and generate 4-5 visualisation
    HM3D_SCENES = get_hm3d_semantic_scenes("data/scene_datasets/hm3d")
    for i, scene in tqdm(enumerate(list(HM3D_SCENES[split]))):
        scene_key = os.path.basename(scene).split(".")[0]
        cfg = get_objnav_config(i, scene_key)
        sim = get_simulator(cfg)
        objects_info = sim.semantic_scene.objects
        pose_sampler = PoseSampler(
            sim=sim,
            r_min=0.1,
            r_max=2.0,
            r_step=0.25,
            rot_deg_delta=10.0,
            h_min=0.8,
            h_max=1.4,
            sample_lookat_deg_delta=5.0,
        )
        print(f"Starting scene: {scene_key}")
        get_relationships_from2d(
            sim, objects_info, scene_key, pose_sampler, relationship_map
        )
        sim.close()


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
