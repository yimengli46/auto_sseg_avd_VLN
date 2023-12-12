import collections
import os
import networkx as nx
import numpy as np
import numpy.linalg as LA
import scipy.io as sio
import cv2
import math
from math import cos, sin, acos, atan2, pi, floor, tan
from io import StringIO
import matplotlib.pyplot as plt
from .constants import colormap
import matplotlib as mpl
# from core import cfg
import png
import json

cfg_SENSOR_AGENT_HEIGHT = 1.25  # 1.25
cfg_SENSOR_DEPTH_MIN = 0.5
cfg_SENSOR_DEPTH_MAX = 8.0
cfg_SEM_MAP_CELL_SIZE = 0.05


def wrap_angle(angle):
    """Wrap the angle to be from -pi to pi."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def minus_theta_fn(previous_theta, current_theta):
    """ compute angle current_theta minus angle previous theta."""
    result = current_theta - previous_theta
    if result < -math.pi:
        result += 2 * math.pi
    if result > math.pi:
        result -= 2 * math.pi
    return result


def plus_theta_fn(previous_theta, current_theta):
    """ compute angle current_theta plus angle previous theta."""
    result = current_theta + previous_theta
    if result < -math.pi:
        result += 2 * math.pi
    if result > math.pi:
        result -= 2 * math.pi
    return result


def project_pixels_to_camera_coords(sseg_img,
                                    current_depth,
                                    current_pose,
                                    gap=2,
                                    FOV=90,
                                    cx=320,
                                    cy=240,
                                    resolution_x=640,
                                    resolution_y=480,
                                    ignored_classes=[]):
    """Project pixels in sseg_img into camera frame given depth image current_depth and camera pose current_pose.

XYZ = K.inv((u, v))
"""
    # camera intrinsic matrix
    FOV = 79
    radian = FOV * pi / 180.
    focal_length = cx / tan(radian / 2)
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    inv_K = LA.inv(K)
    # first compute the rotation and translation from current frame to goal frame
    # then compute the transformation matrix from goal frame to current frame
    # thransformation matrix is the camera2's extrinsic matrix
    tx, tz, theta = current_pose
    R = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0],
                  [-sin(theta), 0, cos(theta)]])
    T = np.array([tx, 0, tz])
    transformation_matrix = np.empty((3, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    # build the point matrix
    x = range(0, resolution_x, gap)
    y = range(0, resolution_y, gap)
    xv, yv = np.meshgrid(np.array(x), np.array(y))
    Z = current_depth[yv.flatten(),
                      xv.flatten()].reshape(yv.shape[0], yv.shape[1])
    points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
    points_4d[:, :, 0] = xv
    points_4d[:, :, 1] = yv
    points_4d[:, :, 2] = Z
    points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1))  # 4 x N

    # apply intrinsic matrix
    points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
    points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
    points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

    # transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
    print('points_4d.shape = {}'.format(points_4d.shape))
    points_3d = points_4d[:3, :]
    print('points_3d.shape = {}'.format(points_3d.shape))

    # pick x-row and z-row
    sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()

    # ignore some classes points
    # print('sseg_points.shape = {}'.format(sseg_points.shape))
    for c in ignored_classes:
        good = (sseg_points != c)
        sseg_points = sseg_points[good]
        points_3d = points_3d[:, good]
    # print('after: sseg_points.shape = {}'.format(sseg_points.shape))
    # print('after: points_3d.shape = {}'.format(points_3d.shape))

    return points_3d, sseg_points.astype(int)


def project_pixels_to_world_coords(sseg_img,
                                   current_depth,
                                   current_pose,
                                   gap=2,
                                   FOV=79,
                                   cx=320,
                                   cy=240,
                                   theta_x=0.0,
                                   resolution_x=640,
                                   resolution_y=480,
                                   ignored_classes=[],
                                   sensor_height=cfg_SENSOR_AGENT_HEIGHT):
    """Project pixels in sseg_img into world frame given depth image current_depth and camera pose current_pose.

(u, v) = KRT(XYZ)
"""

    # camera intrinsic matrix
    radian = FOV * pi / 180.
    focal_length = cx / tan(radian / 2)
    K = np.array([[focal_length, 0, cx], [0, focal_length, cy], [0, 0, 1]])
    inv_K = LA.inv(K)
    # first compute the rotation and translation from current frame to goal frame
    # then compute the transformation matrix from goal frame to current frame
    # thransformation matrix is the camera2's extrinsic matrix
    tx, tz, theta = current_pose
    # theta = -(theta + 0.5 * pi)
    # theta = -theta
    R_y = np.array([[cos(theta), 0, sin(theta)], [0, 1, 0],
                    [-sin(theta), 0, cos(theta)]])
    # used when I tilt the camera up/down
    R_x = np.array([[1, 0, 0], [0, cos(theta_x), -sin(theta_x)],
                    [0, sin(theta_x), cos(theta_x)]])
    R = R_y.dot(R_x)
    T = np.array([tx, 0, tz])
    transformation_matrix = np.empty((3, 4))
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    # build the point matrix
    x = range(0, resolution_x, gap)
    y = range(0, resolution_y, gap)
    xv, yv = np.meshgrid(np.array(x), np.array(y))
    Z = current_depth[yv.flatten(),
                      xv.flatten()].reshape(yv.shape[0], yv.shape[1])
    points_4d = np.ones((yv.shape[0], yv.shape[1], 4), np.float32)
    points_4d[:, :, 0] = xv
    points_4d[:, :, 1] = yv
    points_4d[:, :, 2] = Z
    points_4d = np.transpose(points_4d, (2, 0, 1)).reshape((4, -1))  # 4 x N

    # apply intrinsic matrix
    points_4d[[0, 1, 3], :] = inv_K.dot(points_4d[[0, 1, 3], :])
    points_4d[0, :] = points_4d[0, :] * points_4d[2, :]
    points_4d[1, :] = points_4d[1, :] * points_4d[2, :]

    if False:
        points_4d_img = points_4d.reshape(
            (4, yv.shape[0], yv.shape[1])).transpose((1, 2, 0))
        plt.imshow(points_4d_img[:, :, 1])
        plt.show()

    # transform kp1_4d from camera1(current) frame to camera2(goal) frame through transformation matrix
    points_3d = transformation_matrix.dot(points_4d)

    # reverse y-dim and add sensor height
    points_3d[1, :] = points_3d[1, :] * -1 + sensor_height

    if False:
        points_3d_img = points_3d.reshape(
            (3, yv.shape[0], yv.shape[1])).transpose((1, 2, 0))
        plt.imshow(points_3d_img[:, :, 1])
        plt.show()

    # ignore some artifacts points with depth == 0
    depth_points = current_depth[yv.flatten(), xv.flatten()].flatten()
    good = np.logical_and(depth_points > cfg_SENSOR_DEPTH_MIN,
                          depth_points < cfg_SENSOR_DEPTH_MAX)
    # print(f'points_3d.shape = {points_3d.shape}')
    points_3d = points_3d[:, good]
    # print(f'points_3d.shape = {points_3d.shape}')

    # pick x-row and z-row
    sseg_points = sseg_img[yv.flatten(), xv.flatten()].flatten()
    sseg_points = sseg_points[good]

    # ignore some classes points
    # print('sseg_points.shape = {}'.format(sseg_points.shape))
    for c in ignored_classes:
        good = (sseg_points != c)
        sseg_points = sseg_points[good]
        points_3d = points_3d[:, good]
    # print('after: sseg_points.shape = {}'.format(sseg_points.shape))
    # print('after: points_3d.shape = {}'.format(points_3d.shape))

    return points_3d, sseg_points.astype(int)


def convertInsSegToSSeg(InsSeg, ins2cat_dict):
    """convert instance segmentation image InsSeg (generated by Habitat Simulator) into Semantic segmentation image SSeg,
    given the mapping from instance to category ins2cat_dict.
    """
    ins_id_list = list(ins2cat_dict.keys())
    SSeg = np.zeros(InsSeg.shape, dtype=np.int16)
    for ins_id in ins_id_list:
        SSeg = np.where(InsSeg == ins_id, ins2cat_dict[ins_id], SSeg)
    return SSeg


# if # of classes is <= 41, flag_small_categories is True
def apply_color_to_map(semantic_map, type_categories='SemAVD'):
    """ convert semantic map semantic_map into a colorful visualization color_semantic_map"""
    assert len(semantic_map.shape) == 2

    COLOR = colormap(rgb=True)
    num_classes = 1600

    H, W = semantic_map.shape
    color_semantic_map = np.zeros((H, W, 3), dtype='uint8')
    for i in range(num_classes):
        color_semantic_map[semantic_map == i] = COLOR[i % len(COLOR), 0:3]

    return color_semantic_map


def create_folder(folder_name, clean_up=False):
    """ create folder with directory folder_name.

    If the folder exists before creation, setup clean_up to True to remove files in the folder.
    """
    flag_exist = os.path.isdir(folder_name)
    if not flag_exist:
        print('{} folder does not exist, so create one.'.format(folder_name))
        os.makedirs(folder_name)
        # os.makedirs(os.path.join(test_case_folder, 'observations'))
    else:
        print('{} folder already exists, so do nothing.'.format(folder_name))
        if clean_up:
            os.system('rm {}/*.png'.format(folder_name))
            os.system('rm {}/*.npy'.format(folder_name))
            os.system('rm {}/*.jpg'.format(folder_name))


def read_map_npy(map_npy):
    """ read saved semantic map numpy file infomation."""
    min_x = map_npy['min_x']
    max_x = map_npy['max_x']
    min_z = map_npy['min_z']
    max_z = map_npy['max_z']
    min_X = map_npy['min_X']
    max_X = map_npy['max_X']
    min_Z = map_npy['min_Z']
    max_Z = map_npy['max_Z']
    W = map_npy['W']
    H = map_npy['H']
    semantic_map = map_npy['semantic_map']
    return semantic_map, (min_X, min_Z, max_X, max_Z), (min_x, min_z, max_x,
                                                        max_z), (W, H)


def read_occ_map_npy(map_npy):
    """ read saved occupancy map numpy file infomation."""
    min_x = map_npy['min_x']
    max_x = map_npy['max_x']
    min_z = map_npy['min_z']
    max_z = map_npy['max_z']
    min_X = map_npy['min_X']
    max_X = map_npy['max_X']
    min_Z = map_npy['min_Z']
    max_Z = map_npy['max_Z']
    occ_map = map_npy['occupancy']
    W = map_npy['W']
    H = map_npy['H']
    return occ_map, (min_X, min_Z, max_X, max_Z), (min_x, min_z, max_x,
                                                   max_z), (W, H)


def semanticMap_to_binary(sem_map):
    """ convert semantic map to type 'int8' """
    sem_map.astype('uint8')
    sem_map[sem_map != 2] = 0
    sem_map[sem_map == 2] = 255
    return sem_map


def get_class_mapper(dataset='gibson'):
    """ generate the mapping from category to category idx for dataset Gibson 'gibson' and MP3D dataset as 'mp3d'"""
    class_dict = {}
    return class_dict


def pxl_coords_to_pose(coords,
                       pose_range,
                       coords_range,
                       WH,
                       cell_size=cfg_SEM_MAP_CELL_SIZE,
                       flag_cropped=True):
    """convert cell location 'coords' on the map to pose (X, Z) in the habitat environment"""
    x, y = coords
    min_X, min_Z, max_X, max_Z = pose_range
    min_x, min_z, max_x, max_z = coords_range

    if flag_cropped:
        X = (x + cell_size/2 + min_x) * cell_size + min_X
        Z = (WH[0] - (y + cell_size/2 + min_z)) * cell_size + min_Z
    else:
        X = (x + cell_size/2) * cell_size + min_X
        Z = (WH[0] - (y + cell_size/2)) * cell_size + min_Z
    return (X, Z)


def pose_to_coords(cur_pose,
                   pose_range,
                   coords_range,
                   WH,
                   cell_size=cfg_SEM_MAP_CELL_SIZE,
                   flag_cropped=True):
    """convert pose (X, Z) in the habitat environment to the cell location 'coords' on the map"""
    tx, tz = cur_pose[:2]

    if flag_cropped:
        x_coord = floor((tx - pose_range[0]) / cell_size - coords_range[0])
        z_coord = floor((WH[0] - (tz - pose_range[1]) / cell_size) -
                        coords_range[1])
    else:
        x_coord = floor((tx - pose_range[0]) / cell_size)
        z_coord = floor(WH[0] - (tz - pose_range[1]) / cell_size)

    return (x_coord, z_coord)


def save_sem_map_through_plt(img, name):
    """ save the figure img at directory 'name' using matplotlib"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    # plt.show()
    fig.savefig(name)
    plt.close()


def save_occ_map_through_plt(img, name):
    """ save the figure img at directory 'name' using matplotlib"""
    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img, cmap='gray')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    fig.tight_layout()
    # plt.show()
    fig.savefig(name)
    plt.close()


def gen_arrow_head_marker(rot):
    """generate a marker to plot with matplotlib scatter, plot, ...

    https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers

    rot=0: positive x direction
    Parameters
    ----------
    rot : float
            rotation in radian
            0 is positive x direction

    Returns
    -------
    arrow_head_marker : Path
            use this path for marker argument of plt.scatter
    scale : float
            multiply a argument of plt.scatter with this factor got get markers
            with the same size independent of their rotation.
            Paths are autoscaled to a box of size -1 <= x, y <= 1 by plt.scatter
    """

    # rotate the rot to the marker's coordinate system
    rotate_rot = -(rot - .5 * pi)
    # print(f'rot in drawing is {math.degrees(rot)}, rotate_rot is {math.degrees(rotate_rot)}')
    rot = math.degrees(rotate_rot)
    # print(f'visualized angle = {rot}')

    arr = np.array([[.1, .3], [.1, -.3], [1, 0]])  # arrow shape
    angle = rot / 180 * np.pi
    rot_mat = np.array([[np.cos(angle), np.sin(angle)],
                        [-np.sin(angle), np.cos(angle)]])
    arr = np.matmul(arr, rot_mat)  # rotates the arrow

    # scale
    x0 = np.amin(arr[:, 0])
    x1 = np.amax(arr[:, 0])
    y0 = np.amin(arr[:, 1])
    y1 = np.amax(arr[:, 1])
    scale = np.amax(np.abs([x0, x1, y0, y1]))

    arrow_head_marker = mpl.path.Path(arr)
    return arrow_head_marker, scale


def map_rot_to_planner_rot(rot):
    """ convert rotation on the map 'rot' to the rotation on the environment 'rotate_rot'"""
    rotate_rot = -rot + .5 * pi
    return rotate_rot


def planner_rot_to_map_rot(rot):
    """ convert rotation on the environment 'rot' to rotation on the map 'rotate_rot'"""
    rotate_rot = -(rot - .5 * pi)
    return rotate_rot


def get_img_coordinates(img):
    H, W = img.shape[:2]
    x = np.linspace(0, W - 1, W)
    y = np.linspace(0, H - 1, H)
    xv, yv = np.meshgrid(x, y)
    coords = np.stack((xv, yv), axis=2)
    return coords


def read_all_poses(dataset_root, world):
    """Reads all the poses for each world.

    Args:
      dataset_root: the path to the root of the dataset.
      world: string, name of the world.

    Returns:
      Dictionary of poses for all the images in each world. The key is the image
      id of each view and the values are tuple of (x, z, R, scale). Where x and z
      are the first and third coordinate of translation. R is the 3x3 rotation
      matrix and scale is a float scalar that indicates the scale that needs to
      be multipled to x and z in order to get the real world coordinates.

    Raises:
      ValueError: if the number of images do not match the number of poses read.
    """
    path = os.path.join(dataset_root, world, 'image_structs.mat')
    data = sio.loadmat(path)

    xyz = data['image_structs']['world_pos']
    image_names = data['image_structs']['image_name'][0]
    rot = data['image_structs']['R'][0]
    scale = data['scale'][0][0]
    n = xyz.shape[1]
    x = [xyz[0][i][0][0] for i in range(n)]
    z = [xyz[0][i][2][0] for i in range(n)]
    names = [name[0][:-4] for name in image_names]
    if len(names) != len(x):
        raise ValueError('number of image names are not equal to the number of '
                         'poses {} != {}'.format(len(names), len(x)))
    output = {}
    for i in range(n):
        if rot[i].shape[0] != 0:
            assert rot[i].shape[0] == 3
            assert rot[i].shape[1] == 3
            output[names[i]] = (x[i], z[i], rot[i], scale)
        else:
            output[names[i]] = (x[i], z[i], None, scale)

    return output


def readDepthImage(current_world, current_img_id, AVD_dir, scale):
    img_id = current_img_id[:-1] + '3'
    reader = png.Reader('{}/{}/high_res_depth/{}.png'.format(AVD_dir, current_world, img_id))
    data = reader.asDirect()
    pixels = data[2]
    image = []
    for row in pixels:
        row = np.asarray(row)
        image.append(row)
    image = np.stack(image, axis=0)
    image = image.astype(np.float32)
    image = image / scale
    '''
    if resolution > 0:
        depth = cv2.resize(image, (resolution, resolution), interpolation=cv2.INTER_NEAREST)
    else:
        depth = image
    '''
    return image


def cameraPose2currentPose(current_img_id, camera_pose, image_structs):
    current_x = camera_pose[0]
    current_z = camera_pose[1]
    for i in range(image_structs.shape[0]):
        if image_structs[i][0].item()[:-4] == current_img_id:
            direction = image_structs[i][4]
            break
    current_theta = atan2(direction[2], direction[0])
    current_theta = minus_theta_fn(current_theta, pi / 2)
    current_pose = [current_x, current_z, current_theta]
    return current_pose, direction


def read_cached_data(should_load_images, dataset_root, targets_file_name, output_size, Home_name):
    """Reads all the necessary cached data.

    Args:
      should_load_images: whether to load the images or not.
      dataset_root: path to the root of the dataset.
      segmentation_file_name: The name of the file that contains semantic
        segmentation annotations.
      targets_file_name: The name of the file the contains targets annotated for
        each world.
      output_size: Size of the output images. This is used for pre-processing the
        loaded images.
    Returns:
      Dictionary of all the cached data.
    """

    result_data = {}

    if should_load_images:
        image_path = os.path.join(dataset_root, 'Meta/imgs.npy')
        # loading imgs
        image_data = np.load(image_path, encoding='bytes', allow_pickle=True).item()
        result_data['IMAGE'] = image_data[Home_name]

    word_id_dict_path = os.path.join(dataset_root, 'Meta/world_id_dict.npy')
    result_data['world_id_dict'] = np.load(word_id_dict_path, encoding='bytes', allow_pickle=True).item()

    return result_data


_Graph = collections.namedtuple('_Graph', ['graph', 'id_to_index', 'index_to_id'])


class ActiveVisionDatasetEnv():
    def __init__(self, image_list, current_world, dataset_root):
        self._episode_length = 50
        self._cur_graph = None  # Loaded by _update_graph
        self._world_image_list = image_list
        self._actions = ['right', 'rotate_cw', 'rotate_ccw', 'forward', 'left', 'backward', 'stop']
        # load json file
        f = open('{}/{}/annotations.json'.format(dataset_root, current_world))
        file_content = f.read()
        file_content = file_content.replace('.jpg', '')
        io = StringIO(file_content)
        self._all_graph = json.load(io)
        f.close()

        self._update_graph()

    def to_image_id(self, vid):
        """Converts vertex id to the image id.

        Args:
          vid: vertex id of the view.
        Returns:
          image id of the input vertex id.
        """
        return self._cur_graph.index_to_id[vid]

    def to_vertex(self, image_id):
        return self._cur_graph.id_to_index[image_id]

    def _next_image(self, image_id, action):
        """Given the action, returns the name of the image that agent ends up in.
        Args:
          image_id: The image id of the current view.
          action: valid actions are ['right', 'rotate_cw', 'rotate_ccw',
          'forward', 'left']. Each rotation is 30 degrees.

        Returns:
          The image name for the next location of the agent. If the action results
          in collision or it is not possible for the agent to execute that action,
          returns empty string.
        """
        return self._all_graph[image_id][action]

    def action(self, from_index, to_index):
        return self._cur_graph.graph[from_index][to_index]['action']

    def _update_graph(self):
        """Creates the graph for each environment and updates the _cur_graph."""
        graph = nx.DiGraph()
        id_to_index = {}
        index_to_id = {}
        image_list = self._world_image_list
        for i, image_id in enumerate(image_list):
            image_id = image_id.decode()
            id_to_index[image_id] = i
            index_to_id[i] = image_id
            graph.add_node(i)

        for image_id in image_list:
            image_id = image_id.decode()
            for action in self._actions:
                if action == 'stop' or action == 'left' or action == 'right' or action == 'backward':
                    continue
                next_image = self._all_graph[image_id][action]
                if next_image:
                    graph.add_edge(id_to_index[image_id], id_to_index[next_image], action=action)

                    '''
          current_img = cached_data['IMAGE'][image_id.encode()]
          next_img = cached_data['IMAGE'][next_image.encode()]

          fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10,6))
          ax[0].imshow(current_img)
          ax[0].get_xaxis().set_visible(False)
          ax[0].get_yaxis().set_visible(False)
          ax[0].set_title("current_img")
          ax[1].imshow(next_img)
          ax[1].get_xaxis().set_visible(False)
          ax[1].get_yaxis().set_visible(False)
          ax[1].set_title("next_img")

          fig.tight_layout()
          #fig.savefig('{}/img_{}_proposal_{}.jpg'.format(saved_folder, i, j))
          #plt.close()
          fig.suptitle(f'action = {action}, current_id = {image_id}, next_id = {next_image}')
          plt.show()
          '''

        self._cur_graph = _Graph(graph, id_to_index, index_to_id)
