# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import torch
import torchvision
from PIL import Image
import numpy as np
import os
import pyquaternion
import imageio

from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations

import pdb

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def img_transform(img, post_rot, post_tran,
                  resize, resize_dims, crop,
                  flip, rotate):
    # adjust image
    img = img.resize(resize_dims)
    img = img.crop(crop)
    if flip:
        img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    img = img.rotate(rotate)

    # post-homography transformation
    post_rot *= resize
    post_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b
    A = get_rot(rotate/180*np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    post_rot = A.matmul(post_rot)
    post_tran = A.matmul(post_tran) + b

    return img, post_rot, post_tran


normalize_img = torchvision.transforms.Compose((
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
))


def get_corruption_path(corruption_root, corruption, severity, filepath):
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    # mmcv.mkdir_or_exist(os.path.join(corruption_root, corruption, SEVERITY[str(severity)], subfolder))
    return os.path.join(corruption_root, corruption, severity, subfolder, filename)

@PIPELINES.register_module()
class Custom_LoadMultiViewImageFromFiles_MTL(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, is_train=False, using_ego=False, temporal_consist=False,
                 data_aug_conf={
                     'resize_lim': (0.193, 0.225),
                     'final_dim': (128, 352),
                     'rot_lim': (-5.4, 5.4),
                     'H': 900, 'W': 1600,
                     'rand_flip': True,
                     'bot_pct_lim': (0.0, 0.22),
                     'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                              'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
                     'Ncams': 5,
                 }, load_seg_gt=False, num_seg_classes=14, select_classes=None,
                 corruption=None, severity=None, corruption_root=None):

        self.is_train = is_train
        self.using_ego = using_ego
        self.data_aug_conf = data_aug_conf
        self.load_seg_gt = load_seg_gt
        self.num_seg_classes = num_seg_classes
        self.select_classes = range(
            num_seg_classes) if select_classes is None else select_classes

        self.temporal_consist = temporal_consist
        self.test_time_augmentation = self.data_aug_conf.get('test_aug', False)
        self.corruption = corruption
        self.severity = severity
        self.corruption_root = corruption_root
        if corruption is not None:
            assert severity in ['easy', 'mid', 'hard'], f"Specify a severity of corruption benchmark, now {severity}"
            assert corruption_root is not None, f"When benchmark corruption, specify nuScenes-C root"

    def sample_augmentation(self, specify_resize=None, specify_flip=None):
        H, W = self.data_aug_conf['H'], self.data_aug_conf['W']
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims

            crop_h = max(0, newH - fH)
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)

            flip = False
            if self.data_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
            rotate = np.random.uniform(*self.data_aug_conf['rot_lim'])
        else:
            resize = max(fH / H, fW / W)
            resize = resize + 0.04
            if specify_resize is not None:
                resize = specify_resize

            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = max(0, newH - fH)
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if specify_flip is None else specify_flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def choose_cams(self):
        if self.is_train and self.data_aug_conf['Ncams'] < len(self.data_aug_conf['cams']):
            cams = np.random.choice(self.data_aug_conf['cams'], self.data_aug_conf['Ncams'],
                                    replace=False)
        else:
            cams = self.data_aug_conf['cams']
            cyclist = self.data_aug_conf.get('cyclist', False)
            if cyclist:
                start_id = np.random.choice(np.arange(len(cams)))
                cams = cams[start_id:] + cams[:start_id]
        return cams

    def get_img_inputs(self, results, specify_resize=None, specify_flip=None):
        img_infos = results['img_info']

        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []

        cams = self.choose_cams()
        if self.temporal_consist:
            cam_augments = {}
            for cam in cams:
                cam_augments[cam] = self.sample_augmentation(
                    specify_resize=specify_resize, specify_flip=specify_flip)

        for frame_id, img_info in enumerate(img_infos):
            imgs.append([])
            rots.append([])
            trans.append([])
            intrins.append([])
            post_rots.append([])
            post_trans.append([])

            for cam in cams:
                cam_data = img_info[cam]
                filename = cam_data['data_path']
                # filename = os.path.join(
                #     results['data_root'], filename.split('nuscenes/')[1])
                
                folder, filename = os.path.split(filename)
                subfolder = os.path.split(folder)[1]
                filename = os.path.join(subfolder, filename)
                filename = get_corruption_path(self.corruption_root, self.corruption, self.severity, filename)

                img = Image.open(filename)

                # img = imageio.imread(filename)
                # img = Image.fromarray(img)

                post_rot = torch.eye(2)
                post_tran = torch.zeros(2)

                intrin = torch.Tensor(cam_data['cam_intrinsic'])
                # extrinsics
                rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
                tran = torch.Tensor(cam_data['sensor2lidar_translation'])

                # 进一步转换到 LiDAR 坐标系
                if self.using_ego:
                    cam2lidar = torch.eye(4)
                    cam2lidar[:3, :3] = torch.Tensor(
                        cam_data['sensor2lidar_rotation'])
                    cam2lidar[:3, 3] = torch.Tensor(
                        cam_data['sensor2lidar_translation'])

                    lidar2ego = torch.eye(4)
                    lidar2ego[:3, :3] = results['lidar2ego_rots']
                    lidar2ego[:3, 3] = results['lidar2ego_trans']

                    cam2ego = lidar2ego @ cam2lidar

                    rot = cam2ego[:3, :3]
                    tran = cam2ego[:3, 3]

                # augmentation (resize, crop, horizontal flip, rotate)
                if self.temporal_consist:
                    resize, resize_dims, crop, flip, rotate = cam_augments[cam]
                else:
                    # generate augmentation for each time-step, each
                    resize, resize_dims, crop, flip, rotate = self.sample_augmentation(
                        specify_resize=specify_resize, specify_flip=specify_flip)

                img, post_rot2, post_tran2 = img_transform(img, post_rot, post_tran,
                                                           resize=resize,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)

                post_tran = torch.zeros(3)
                post_rot = torch.eye(3)
                post_tran[:2] = post_tran2
                post_rot[:2, :2] = post_rot2

                imgs[frame_id].append(normalize_img(img))
                intrins[frame_id].append(intrin)
                rots[frame_id].append(rot)
                trans[frame_id].append(tran)
                post_rots[frame_id].append(post_rot)
                post_trans[frame_id].append(post_tran)

        # [num_seq, num_cam, ...]
        imgs = torch.stack([torch.stack(x, dim=0) for x in imgs], dim=0)
        rots = torch.stack([torch.stack(x, dim=0) for x in rots], dim=0)
        trans = torch.stack([torch.stack(x, dim=0) for x in trans], dim=0)
        intrins = torch.stack([torch.stack(x, dim=0) for x in intrins], dim=0)
        post_rots = torch.stack([torch.stack(x, dim=0)
                                for x in post_rots], dim=0)
        post_trans = torch.stack([torch.stack(x, dim=0)
                                 for x in post_trans], dim=0)

        return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        if (not self.is_train) and self.test_time_augmentation:
            results['flip_aug'] = []
            results['scale_aug'] = []
            img_inputs = []
            for flip in self.data_aug_conf.get('tta_flip', [False, ]):
                for scale in self.data_aug_conf.get('tta_scale', [None, ]):
                    results['flip_aug'].append(flip)
                    results['scale_aug'].append(scale)
                    img_inputs.append(
                        self.get_img_inputs(results, scale, flip))

            results['img_inputs'] = img_inputs
        else:
            results['img_inputs'] = self.get_img_inputs(results)

        return results