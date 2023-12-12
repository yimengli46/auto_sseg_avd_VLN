import numpy as np
from math import pi
from utils.baseline_utils import convertInsSegToSSeg, create_folder, read_all_poses, readDepthImage, cameraPose2currentPose
from utils.build_map_utils import SemanticMap
import random
# from core import cfg
import json
import os
import cv2
import matplotlib.pyplot as plt
import scipy.io as sio

# =========================================== fix the habitat scene shuffle ===============================
SEED = 5
random.seed(SEED)
np.random.seed(SEED)

output_folder = f'output/semantic_map'
# after testing, using 8 angles is most efficient
theta_lst = [0, pi/4, pi/2, pi*3./4, pi, pi*5./4, pi*3./2, pi*7./4]
str_theta_lst = ['000', '090', '180', '270']

cfg_SEM_MAP_WORLD_SIZE = 50.0

# ============================= build a grid =========================================
x = np.arange(-cfg_SEM_MAP_WORLD_SIZE, cfg_SEM_MAP_WORLD_SIZE, 0.3)
z = np.arange(-cfg_SEM_MAP_WORLD_SIZE, cfg_SEM_MAP_WORLD_SIZE, 0.3)
xv, zv = np.meshgrid(x, z)
grid_H, grid_W = zv.shape

# ====================================================================================
avd_minimal_dir = '/home/yimeng/ARGO_datasets/Datasets/AVD_Minimal'
sseg_dir = '/home/yimeng/ARGO_scratch/auto_sseg_avd/sseg_sam/output/stage_f_sem_seg'
scenes = ['Home_001_1']
scenes = ['Home_007_1', 'Home_008_1',
          'Home_010_1', 'Home_011_1', 'Home_013_1', 'Home_014_1', 'Home_014_2', 'Home_015_1',
          'Home_016_1', 'Office_001_1']

for scene_id in range(len(scenes)):
    print(f'scene_id = {scene_id}')
    scene_name = scenes[scene_id]

    saved_folder = f'{output_folder}/{scene_name}'
    create_folder(saved_folder, clean_up=False)

    npy_file = f'{saved_folder}/BEV_semantic_map.npy'

    # key: img_name, val: (x, z, rot, scale)
    all_poses = read_all_poses(avd_minimal_dir, scene_name)
    # load img structs
    image_structs_path = os.path.join(f'{avd_minimal_dir}/{scene_name}', 'image_structs.mat')
    image_structs = sio.loadmat(image_structs_path)
    image_structs = image_structs['image_structs']
    image_structs = image_structs[0]

    # ================================ Building a map ===============================
    SemMap = SemanticMap(saved_folder)

    count_ = 0
    # ========================= generate observations ===========================
    for img_id in list(all_poses.keys()):  # ['000110000030101', '000110000270101']:  # list(all_poses.keys()):
        print(f'img_id = {img_id}')

        # load images, depth, semantic_seg and pose
        camera_pose = all_poses[img_id]  # x, z, R, f
        scale = camera_pose[3]

        try:
            rgb_img = cv2.imread(f'{avd_minimal_dir}/{scene_name}/jpg_rgb/{img_id}.jpg', 1)[:, :, ::-1]
            depth_img = readDepthImage(scene_name, img_id, avd_minimal_dir, scale)
            sseg_img = cv2.imread(f'{sseg_dir}/{scene_name}/{img_id}_labels.png',
                                  cv2.IMREAD_UNCHANGED).astype(np.uint16)
        except:
            continue

        current_pose, _ = cameraPose2currentPose(img_id, camera_pose, image_structs)

        pose = current_pose

        SemMap.build_semantic_map(
            rgb_img, depth_img, sseg_img, pose, count_)
        count_ += 1

    SemMap.save_final_map()
