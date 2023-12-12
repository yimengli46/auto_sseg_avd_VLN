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
import torch
import clip


def transfer_coord_system(img, pose_range, coords_range, WH):
    center_x, center_y = pose_to_coords((0, 0), pose_range, coords_range, WH)
    x1 = 0 - center_x + 500
    z1 = 0 - center_y + 500
    x2 = W - center_x + 500
    z2 = H - center_y + 500
    img = img[z1:z2, x1:x2]
    return img


def get_text_feats(in_text, clip_model, clip_feat_dim, batch_size=64):
    text_tokens = clip.tokenize(in_text).cuda()
    text_id = 0
    text_feats = np.zeros((len(in_text), clip_feat_dim), dtype=np.float32)
    while text_id < len(text_tokens):  # Batched inference.
        batch_size = min(len(in_text) - text_id, batch_size)
        text_batch = text_tokens[text_id: text_id + batch_size]
        with torch.no_grad():
            batch_feats = clip_model.encode_text(text_batch).float()
        batch_feats /= batch_feats.norm(dim=-1, keepdim=True)
        batch_feats = np.float32(batch_feats.cpu())
        text_feats[text_id: text_id + batch_size, :] = batch_feats
        text_id += batch_size
    return text_feats


scenes = ['Home_001_1']

# scene_list = ['Home_001_1', 'Home_002_1', 'Home_003_1', 'Home_004_1', 'Home_005_1', 'Home_006_1',
#               'Home_007_1', 'Home_008_1', 'Home_010_1', 'Home_011_1', 'Home_014_1', 'Home_014_2',
#               'Home_015_1', 'Home_016_1']

sem_map_folder = f'output/semantic_map'
avd_minimal_dir = '/home/yimeng/ARGO_datasets/Datasets/AVD_Minimal'
vl_map_folder = f'/home/yimeng/ARGO_scratch/vlmaps/data_AVD'

device = "cuda" if torch.cuda.is_available() else "cpu"
clip_version = "ViT-B/32"
clip_feat_dim = {'RN50': 1024, 'RN101': 512, 'RN50x4': 640, 'RN50x16': 768,
                 'RN50x64': 1024, 'ViT-B/32': 512, 'ViT-B/16': 512, 'ViT-L/14': 768}[clip_version]
clip_model, preprocess = clip.load(clip_version)  # clip.available_models()
clip_model.to(device).eval()


for scene_id in range(len(scenes)):
    scene_name = scenes[scene_id]
    print(f'scene_name = {scene_name}')

    # =============== load vl_map rgb
    rgb_vl_map_path = f'{vl_map_folder}/{scene_name}/map/color_top_down_1.npy'
    rgb_vl_map = np.load(rgb_vl_map_path)

    clip_vl_map_path = f'{vl_map_folder}/{scene_name}/map/grid_lseg_1.npy'
    clip_vl_map = np.load(clip_vl_map_path)

    # ====================== load the semantic map
    sem_map_npy = np.load(f'output/semantic_map/{scene_name}/BEV_semantic_map.npy', allow_pickle=True).item()
    semantic_map, pose_range, coords_range, WH = read_map_npy(sem_map_npy)
    H, W = semantic_map.shape

    color_semantic_map = apply_color_to_map(semantic_map)

    # ================= transfer vl_map to the same coordinate system as my built semantic map
    rgb_vl_map = transfer_coord_system(rgb_vl_map, pose_range, coords_range, WH)
    clip_vl_map = transfer_coord_system(clip_vl_map, pose_range, coords_range, WH)

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
    ax.imshow(rgb_vl_map)
    fig.tight_layout()

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

    # ====== find the pose closest to the target object
    # target_object_id = 185  # handbag
    target_object_name = 'handbag'

    # query vl_map for the target location
    text_feats = get_text_feats([target_object_name], clip_model, clip_feat_dim)  # N x 512
    N = text_feats.shape[0]
    map_feats = clip_vl_map.reshape((-1, clip_vl_map.shape[-1]))
    scores_list = map_feats @ text_feats.T
    scores_all = scores_list.reshape((H, W))  # H x W

    # normalize the score
    min_value = np.min(scores_all)
    max_value = np.max(scores_all)
    scores_all = (scores_all - min_value) / (max_value - min_value)

    # find the target center with maximum value
    max_index = np.argmax(scores_all)
    # Convert the flattened index to row and column indices
    row_index, col_index = np.unravel_index(max_index, scores_all.shape)
    print(f'max scores = {scores_all[row_index, col_index]}')
    target_center = (col_index, row_index)

    # visualize the score
    ax.imshow(scores_all)
    ax.plot(col_index, row_index, color='r', marker='*', markersize=20)

    # find the nearest img to the target center
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

    plt.show()
