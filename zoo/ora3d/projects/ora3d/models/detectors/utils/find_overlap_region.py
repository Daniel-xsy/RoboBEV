import torch
import numpy as np


def find_overlap_left(cam2lidar_1, lidar2cam_2):
    infinity = 10000

    # Camera coordinate system representation of the top left and bottom left points of the image
    cam_left_top_1 = torch.FloatTensor([0.001 * infinity, 0.001 * infinity, infinity, 1])
    cam_left_bottom_1 = torch.FloatTensor([0.001 * infinity, 899.999 * infinity, infinity, 1])

    # Convert a point in the camera coordinate system to a point in the lidar coordinate system
    cam2lidar_left_top_1 = torch.matmul(cam2lidar_1, cam_left_top_1)
    cam2lidar_left_bottom_1 = torch.matmul(cam2lidar_1, cam_left_bottom_1)

    # Convert a point in the lidar coordinate system to a point in another camera coordinate system
    cam_left_top_2 = torch.matmul(lidar2cam_2, cam2lidar_left_top_1)
    cam_left_bottom_2 = torch.matmul(lidar2cam_2, cam2lidar_left_bottom_1)

    # Convert a point in the camera coordinate system to a point on a 2D image coordinate system
    img_left_top_2 = cam_left_top_2 / cam_left_top_2[2]
    img_left_bottom_2 = cam_left_bottom_2 / cam_left_bottom_2[2]

    return img_left_top_2, img_left_bottom_2


def find_overlap_right(cam2lidar_1, lidar2cam_2):
    infinity = 10000

    cam_right_top_1 = torch.FloatTensor([1599.999 * infinity, 0.001 * infinity, infinity, 1])
    cam_right_bottom_1 = torch.FloatTensor([1599.999 * infinity, 899.999 * infinity, infinity, 1])

    cam2lidar_right_top_1 = torch.matmul(cam2lidar_1, cam_right_top_1)
    cam2lidar_right_bottom_1 = torch.matmul(cam2lidar_1, cam_right_bottom_1)

    cam_right_top_2 = torch.matmul(lidar2cam_2, cam2lidar_right_top_1)
    cam_right_bottom_2 = torch.matmul(lidar2cam_2, cam2lidar_right_bottom_1)

    img_right_top_2 = cam_right_top_2 / cam_right_top_2[2]
    img_right_bottom_2 = cam_right_bottom_2 / cam_right_bottom_2[2]

    return img_right_top_2, img_right_bottom_2


def find_overlap_region(lidar2cam):
    lidar2cam_dict = {
        'front': torch.tensor(np.float32(lidar2cam[0][0])),
        'front_right': torch.tensor(np.float32(lidar2cam[0][1])),
        'front_left': torch.tensor(np.float32(lidar2cam[0][2])),
        'back': torch.tensor(np.float32(lidar2cam[0][3])),
        'back_left': torch.tensor(np.float32(lidar2cam[0][4])),
        'back_right': torch.tensor(np.float32(lidar2cam[0][5]))
    }

    cam2lidar_dict = {
        'front': torch.inverse(lidar2cam_dict['front']),
        'front_right': torch.inverse(lidar2cam_dict['front_right']),
        'front_left': torch.inverse(lidar2cam_dict['front_left']),
        'back': torch.inverse(lidar2cam_dict['back']),
        'back_left': torch.inverse(lidar2cam_dict['back_left']),
        'back_right': torch.inverse(lidar2cam_dict['back_right'])
    }

    overlap_match_list = [
        ['front_right', 'front'], ['back_right', 'front_right'], ['front', 'front_left'],
        ['back_left', 'back'], ['front_left', 'back_left'], ['back', 'back_right']
    ]

    overlap_region_result = list()

    for cam_pair in overlap_match_list:
        lidar2cam_A = lidar2cam_dict[cam_pair[0]]
        lidar2cam_B = lidar2cam_dict[cam_pair[1]]
        cam2lidar_A = cam2lidar_dict[cam_pair[0]]
        cam2lidar_B = cam2lidar_dict[cam_pair[1]]

        left_overlap = find_overlap_left(cam2lidar_A, lidar2cam_B)
        right_overlap = find_overlap_right(cam2lidar_B, lidar2cam_A)

        left_line = (int(left_overlap[0][0]), int(left_overlap[0][1]), int(left_overlap[1][0]), int(left_overlap[1][1]))
        right_line = (int(right_overlap[0][0]), int(right_overlap[0][1]), int(right_overlap[1][0]), int(right_overlap[1][1]))

        overlap_region_result.append([left_line, right_line])

    return overlap_region_result


def overlap_region_pts(overlap_regions):
    overlap_regions_pts = list()

    for overlap_region in overlap_regions:
        if overlap_region[0][0] < overlap_region[0][2]:
            ov_start_pt = overlap_region[0][0]
            ov_width = 1600 - overlap_region[0][0]
        else:
            ov_start_pt = overlap_region[0][2]
            ov_width = 1600 - overlap_region[0][2]
        overlap_regions_pts.append((ov_start_pt, ov_width))

    return overlap_regions_pts