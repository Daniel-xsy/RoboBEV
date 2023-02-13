# Copyright (c) OpenMMLab. All rights reserved.
import os
import torch
import torchvision
from PIL import Image
import mmcv
import numpy as np
from pyquaternion import Quaternion

from mmdet3d.core.points import BasePoints, get_points_type
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import LoadAnnotations, LoadImageFromFile


def get_corruption_path(corruption_root, corruption, severity, filepath):
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    # mmcv.mkdir_or_exist(os.path.join(corruption_root, corruption, SEVERITY[str(severity)], subfolder))
    return os.path.join(corruption_root, corruption, severity, subfolder, filename)


@PIPELINES.register_module()
class Custom_LoadMultiViewImageFromFiles_BEVDet(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, data_config, is_train=False,
                 sequential=False, aligned=False, trans_only=True,
                 corruption=None, severity=None, corruption_root=None):
        self.is_train = is_train
        self.data_config = data_config
        self.normalize_img = torchvision.transforms.Compose((
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])))
        self.sequential = sequential
        self.aligned = aligned
        self.trans_only = trans_only

        # Test on nuScenes-c dataset
        self.corruption = corruption
        self.severity = severity
        self.corruption_root = corruption_root
        if corruption is not None:
            assert severity in ['easy', 'mid', 'hard'], f"Specify a severity of corruption benchmark, now {severity}"
            assert corruption_root is not None, f"When benchmark corruption, specify nuScenes-C root"

    def get_rot(self,h):
        return torch.Tensor([
            [np.cos(h), np.sin(h)],
            [-np.sin(h), np.cos(h)],
        ])

    def img_transform(self, img, post_rot, post_tran,
                      resize, resize_dims, crop,
                      flip, rotate):
        # adjust image
        img = self.img_transform_core(img, resize_dims, crop, flip, rotate)

        # post-homography transformation
        post_rot *= resize
        post_tran -= torch.Tensor(crop[:2])
        if flip:
            A = torch.Tensor([[-1, 0], [0, 1]])
            b = torch.Tensor([crop[2] - crop[0], 0])
            post_rot = A.matmul(post_rot)
            post_tran = A.matmul(post_tran) + b
        A = self.get_rot(rotate / 180 * np.pi)
        b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
        b = A.matmul(-b) + b
        post_rot = A.matmul(post_rot)
        post_tran = A.matmul(post_tran) + b

        return img, post_rot, post_tran

    def img_transform_core(self, img, resize_dims, crop, flip, rotate):
        # adjust image
        img = img.resize(resize_dims)
        img = img.crop(crop)
        if flip:
            img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
        img = img.rotate(rotate)
        return img

    def choose_cams(self):
        if self.is_train and self.data_config['Ncams'] < len(self.data_config['cams']):
            cams = np.random.choice(self.data_config['cams'], self.data_config['Ncams'],
                                    replace=False)
        else:
            cams = self.data_config['cams']
        return cams

    def sample_augmentation(self, H , W, flip=None, scale=None):
        fH, fW = self.data_config['input_size']
        if self.is_train:
            resize = float(fW)/float(W)
            resize += np.random.uniform(*self.data_config['resize'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.random.uniform(*self.data_config['crop_h'])) * newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = self.data_config['flip'] and np.random.choice([0, 1])
            rotate = np.random.uniform(*self.data_config['rot'])
        else:
            resize = float(fW)/float(W)
            resize += self.data_config.get('resize_test', 0.0)
            if scale is not None:
                resize = scale
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.data_config['crop_h'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False if flip is None else flip
            rotate = 0
        return resize, resize_dims, crop, flip, rotate

    def get_inputs(self,results, flip=None, scale=None):
        imgs = []
        rots = []
        trans = []
        intrins = []
        post_rots = []
        post_trans = []
        cams = self.choose_cams()
        for cam in cams:
            cam_data = results['img_info'][cam]
            orig_filename = cam_data['data_path']

            filename = os.path.split(orig_filename)[1]
            subfolder = os.path.split(os.path.split(orig_filename)[0])[1]
            filename = os.path.join(subfolder, filename)
            filename = get_corruption_path(self.corruption_root, self.corruption, self.severity, filename)

            img = Image.open(filename)
            post_rot = torch.eye(2)
            post_tran = torch.zeros(2)

            intrin = torch.Tensor(cam_data['cam_intrinsic'])
            rot = torch.Tensor(cam_data['sensor2lidar_rotation'])
            tran = torch.Tensor(cam_data['sensor2lidar_translation'])

            # augmentation (resize, crop, horizontal flip, rotate)
            resize, resize_dims, crop, flip, rotate = self.sample_augmentation(H=img.height,
                                                                               W=img.width,
                                                                               flip=flip,
                                                                               scale=scale)
            img, post_rot2, post_tran2 = self.img_transform(img, post_rot, post_tran,
                                                            resize=resize,
                                                            resize_dims=resize_dims,
                                                            crop=crop,
                                                            flip=flip,
                                                            rotate=rotate)

            # for convenience, make augmentation matrices 3x3
            post_tran = torch.zeros(3)
            post_rot = torch.eye(3)
            post_tran[:2] = post_tran2
            post_rot[:2, :2] = post_rot2

            imgs.append(self.normalize_img(img))

            if self.sequential:
                assert 'adjacent' in results
                if not type(results['adjacent']) is list:
                    filename_adjacent = results['adjacent']['cams'][cam]['data_path']
                    img_adjacent = Image.open(filename_adjacent)
                    img_adjacent = self.img_transform_core(img_adjacent,
                                                           resize_dims=resize_dims,
                                                           crop=crop,
                                                           flip=flip,
                                                           rotate=rotate)
                    imgs.append(self.normalize_img(img_adjacent))
                else:
                    for id in range(len(results['adjacent'])):
                        filename_adjacent = results['adjacent'][id]['cams'][cam]['data_path']
                        img_adjacent = Image.open(filename_adjacent)
                        img_adjacent = self.img_transform_core(img_adjacent,
                                                               resize_dims=resize_dims,
                                                               crop=crop,
                                                               flip=flip,
                                                               rotate=rotate)
                        imgs.append(self.normalize_img(img_adjacent))
            intrins.append(intrin)
            rots.append(rot)
            trans.append(tran)
            post_rots.append(post_rot)
            post_trans.append(post_tran)

        if self.sequential:
            if self.trans_only:
                if not type(results['adjacent']) is list:
                    rots.extend(rots)
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        posi_curr = np.array(results['curr']['ego2global_translation'], dtype=np.float32)
                        posi_adj = np.array(results['adjacent']['ego2global_translation'], dtype=np.float32)
                        shift_global = posi_adj - posi_curr

                        l2e_r = results['curr']['lidar2ego_rotation']
                        e2g_r = results['curr']['ego2global_rotation']
                        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
                        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

                        # shift_global = np.array([*shift_global[:2], 0.0])
                        shift_lidar = shift_global @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                            l2e_r_mat).T
                        trans.extend([tran + shift_lidar for tran in trans])
                    else:
                        trans.extend(trans)
                else:
                    assert False
            else:
                if not type(results['adjacent']) is list:
                    post_trans.extend(post_trans)
                    post_rots.extend(post_rots)
                    intrins.extend(intrins)
                    if self.aligned:
                        egocurr2global = np.eye(4, dtype=np.float32)
                        egocurr2global[:3,:3] = Quaternion(results['curr']['ego2global_rotation']).rotation_matrix
                        egocurr2global[:3,3] = results['curr']['ego2global_translation']

                        egoadj2global = np.eye(4, dtype=np.float32)
                        egoadj2global[:3,:3] = Quaternion(results['adjacent']['ego2global_rotation']).rotation_matrix
                        egoadj2global[:3,3] = results['adjacent']['ego2global_translation']

                        lidar2ego = np.eye(4, dtype=np.float32)
                        lidar2ego[:3, :3] = Quaternion(results['curr']['lidar2ego_rotation']).rotation_matrix
                        lidar2ego[:3, 3] = results['curr']['lidar2ego_translation']

                        lidaradj2lidarcurr = np.linalg.inv(lidar2ego) @ np.linalg.inv(egocurr2global) \
                                             @ egoadj2global @ lidar2ego
                        trans_new = []
                        rots_new =[]
                        for tran,rot in zip(trans, rots):
                            mat = np.eye(4, dtype=np.float32)
                            mat[:3,:3] = rot
                            mat[:3,3] = tran
                            mat = lidaradj2lidarcurr @ mat
                            rots_new.append(torch.from_numpy(mat[:3,:3]))
                            trans_new.append(torch.from_numpy(mat[:3,3]))
                        rots.extend(rots_new)
                        trans.extend(trans_new)
                    else:
                        rots.extend(rots)
                        trans.extend(trans)
                else:
                    assert False
        imgs, rots, trans, intrins, post_rots, post_trans = (torch.stack(imgs), torch.stack(rots), torch.stack(trans),
                                                             torch.stack(intrins), torch.stack(post_rots),
                                                             torch.stack(post_trans))
        return imgs, rots, trans, intrins, post_rots, post_trans

    def __call__(self, results):
        results['img_inputs'] = self.get_inputs(results)
        return results