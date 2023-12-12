import numpy as np
from math import pi
from utils.baseline_utils import create_folder, read_all_poses, readDepthImage, read_map_npy, apply_color_to_map, cameraPose2currentPose, pose_to_coords, read_cached_data, ActiveVisionDatasetEnv
import random
import json
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio
import skimage.measure
from math import floor
import networkx as nx
from categories import dataset_dict

scenes = ['Home_001_1']

# scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
#               'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
#               'Home_015_1', 'Home_016_1']

sem_map_folder = f'output/semantic_map'
avd_minimal_dir = '/home/yimeng/ARGO_datasets/Datasets/AVD_Minimal'

for scene_id in range(len(scenes)):
    scene_name = scenes[scene_id]
    print(f'scene_name = {scene_name}')

    # ====================== load the semantic map
    sem_map_npy = np.load(f'output/semantic_map/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
    semantic_map, pose_range, coords_range, WH = read_map_npy(sem_map_npy)

    color_semantic_map = apply_color_to_map(semantic_map)

    # ============== draw the poses
    all_poses = read_all_poses(avd_minimal_dir, scene_name)
    # load img structs
    image_structs_path = os.path.join(f'{avd_minimal_dir}/{scene_name}', 'image_structs.mat')
    image_structs = sio.loadmat(image_structs_path)
    image_structs = image_structs['image_structs']
    image_structs = image_structs[0]

    cached_data = read_cached_data(False, avd_minimal_dir, targets_file_name=None,
                                   output_size=224, Home_name=scene_name.encode())  # encode() convert string to byte
    current_world_image_ids = cached_data['world_id_dict'][scene_name.encode()]
    # initialize the graph map
    AVD = ActiveVisionDatasetEnv(current_world_image_ids, scene_name, avd_minimal_dir)

    # visualize the semantic map
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 15))
    ax.imshow(color_semantic_map)
    fig.tight_layout()
    ax.imshow(color_semantic_map)
    # ax.get_xaxis().set_visible(False)
    # ax.get_yaxis().set_visible(False)

    all_coords = {}
    for img_id in list(all_poses.keys()):
        # print(f'img_id = {img_id}')

        # load images, depth, semantic_seg and pose
        camera_pose = all_poses[img_id]  # x, z, R, f
        scale = camera_pose[3]

        current_pose, _ = cameraPose2currentPose(img_id, camera_pose, image_structs)

        cur_coords = pose_to_coords(current_pose, pose_range, coords_range, WH)
        # print(f'{cur_coords}')

        x, z = cur_coords
        all_coords[img_id] = cur_coords

        ax.plot(x, z, color='blue', marker='o', markersize=5)

    # ======================= localize all the objects ================================
    H, W = semantic_map.shape
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)

    # compute centers of semantic classes
    print('compute object centers ...')
    cat_binary_map = semantic_map.copy()
    for cat in range(0, 150):
        cat_binary_map = np.where(
            cat_binary_map == cat, -1, cat_binary_map)
    # run skimage to find the number of objects belong to this class
    instance_label, num_ins = skimage.measure.label(
        cat_binary_map, background=-1, connectivity=1, return_num=True)

    # find all the object centroids
    list_instances = []
    for idx_ins in range(1, num_ins + 1):
        mask_ins = (instance_label == idx_ins)
        if np.sum(mask_ins) >= 20:  # should have at least 1 pixels
            x_coords = xv[mask_ins]
            y_coords = yv[mask_ins]
            ins_center = (floor(np.median(x_coords)),
                          floor(np.median(y_coords)))
            ins_cat = semantic_map[int(
                y_coords[0]), int(x_coords[0])]

            ins = {}
            ins['center'] = (ins_center[0],
                             ins_center[1])
            ins['cat'] = ins_cat
            ins['id_ins'] = idx_ins
            ins['mask'] = mask_ins
            list_instances.append(ins)

    # draw the instance names
    x, y = [], []
    for ins in list_instances:
        center = ins['center']
        cat = ins['cat']

        x.append(center[0])
        y.append(center[1])

        try:
            cat_name = dataset_dict[cat]
        except:
            cat_name = 'unknown'
        ax.text(center[0], center[1], cat_name)

    ax.scatter(x=x, y=y, c='w', s=5)

    # ====== find the pose closest to the target object
    target_object_id = 185  # handbag
    for ins_idx, ins in enumerate(list_instances):
        cat = ins['cat']
        if cat == target_object_id:
            break

    target_center = ins['center']

    nearest_idx = -1
    nearest_dist = 1e5
    for img_id in list(all_coords.keys()):
        cur_coords = all_coords[img_id]
        dist = (cur_coords[0] - target_center[0])**2 + (cur_coords[1] - target_center[1])**2
        if dist < nearest_dist:
            nearest_idx = img_id
            nearest_dist = dist

    print(f'nearest img is {nearest_idx}, dist = {nearest_dist}')

    # ================== A* planning
    start_img_id = list(all_poses.keys())[0]
    start_img_vertex = AVD.to_vertex(start_img_id)
    target_img_vertex = AVD.to_vertex(nearest_idx)
    path = nx.shortest_path(AVD._cur_graph.graph, start_img_vertex, target_img_vertex)

    list_visited_img_id = []
    for j in range(0, len(path)):
        img_id = AVD.to_image_id(path[j])
        list_visited_img_id.append(img_id)

    # draw the path:
    xs = []
    zs = []
    for img_id in list_visited_img_id:
        x, z = all_coords[img_id]
        xs.append(x)
        zs.append(z)

    ax.plot(xs, zs, color='black', marker='o', markersize=5)

    plt.show()
