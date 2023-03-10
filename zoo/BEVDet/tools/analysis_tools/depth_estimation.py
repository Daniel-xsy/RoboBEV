# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Shaoyuan Xie
#  Used to visualize depth estimation results
# ---------------------------------------------

import os
import time
import numpy as np
import os.path as osp
import warnings
import argparse
from typing import Tuple,List

import mmcv
import torch

from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)

from mmdet.apis import set_random_seed
from mmdet.datasets import replace_ImageToTensor
import mmdet3d
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

from mmdet3d.datasets import build_dataloader

import matplotlib.pyplot as plt
import matplotlib
import torchvision.transforms.functional as F
from torch.nn.functional import interpolate
from torchvision.utils import make_grid


def denormalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    
    img = img * torch.tensor(std).view(1, 1, 3) + torch.tensor(mean).view(1, 1, 3)
    img = img * 255
    img = img.numpy()
    img = img[...,::-1]

    return img.astype(np.uint8)

def depth_to_image(depth, imgs, camera_idx=1,  H=256, W=704):
    """Convert depth estimation results to gray scale image
    Aegs:
        depth (List): n x [multivew, depth_quant, h, w]
    """
    assert len(depth) == len(imgs)

    depth = [depth_[camera_idx].unsqueeze(dim=0) for depth_ in depth]
    imgs = [img[camera_idx].permute(1, 2, 0) for img in imgs]
    depth = torch.cat(depth, dim=0).argmax(dim=1, keepdim=True).float()   # [N, 1, H, W]
    depth = interpolate(depth, size=(H, W), mode='bilinear', align_corners=False)
    depth = depth.permute(0, 2, 3, 1).cpu().numpy()  # [N, H, W, 1]
    depth = (depth / 59) # * 255.
    # depth = depth.astype(np.uint8)
    for i, depth_ in enumerate(depth):
        # mmcv.imwrite(depth_, f'./tools/figure/test{i}.png')
        mmcv.imwrite(denormalize(imgs[i]), f'./tools/figure/img{i}.png')
        plt.imsave(f'./tools/figure/test{i}.png', np.squeeze(depth_), cmap='plasma')


def calculate_mse(depth):
    """Calculate the MSE between depth estimation under corruptions and clean input
    Args:
        depth (List): n x [multivew, depth_quant, h, w]
    """

depth_estimation = []
def layer_hook(module, inputs, outputs):
    depth_estimation.append(outputs[:, :59].softmax(dim=1))


def show(imgs, mean, std):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = denormalize(img, mean, std)
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet attack a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--attack', action='store_true', default=False)
    parser.add_argument('--show', action='store_true', default=False)
    args = parser.parse_args()

    return args


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                if isinstance(cfg.plugin_dir, str):
                    cfg.plugin_dir = [cfg.plugin_dir]
                # import multi plugin modules
                for plugin_dir_ in cfg.plugin_dir:
                    _module_dir = os.path.dirname(plugin_dir_)
                    _module_dir = _module_dir.split('/')
                    _module_path = _module_dir[0]

                    for m in _module_dir[1:]:
                        _module_path = _module_path + '.' + m
                    print(_module_path)
                    plg_lib = importlib.import_module(_module_path)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    cfg.model.pretrained = None
    # in case the test dataset is concatenated
    samples_per_gpu = 1
    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        samples_per_gpu = cfg.data.test.pop('samples_per_gpu', 1)
        if samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        samples_per_gpu = max(
            [ds_cfg.pop('samples_per_gpu', 1) for ds_cfg in cfg.data.test])
        if samples_per_gpu > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    # test_mode false to return ground truth used in the attack
    # chnage after build the dataset to avoid data filtering
    # an ugly workaround to set test_mode = False
    # dataset.test_mode = False
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    # old versions did not save class info in checkpoints, this walkaround is
    # for backward compatibility
    if 'CLASSES' in checkpoint.get('meta', {}):
        model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        model.CLASSES = dataset.CLASSES
    # palette for visualization in segmentation tasks
    if 'PALETTE' in checkpoint.get('meta', {}):
        model.PALETTE = checkpoint['meta']['PALETTE']
    elif hasattr(dataset, 'PALETTE'):
        # segmentation dataset has `PALETTE` attribute
        model.PALETTE = dataset.PALETTE

    for n, p in model.named_parameters():
        p.requires_grad = False
    model = MMDataParallel(model, device_ids=[0])
    hook = model.module.img_view_transformer.depthnet.register_forward_hook(layer_hook)

    iteration = 22

    imgs = []
    data_loader.dataset.pipeline.transforms[0].corruption = 'Clean'

    data_loader_ = iter(data_loader)
    for i in range(iteration):
        data = next(data_loader_)
    imgs.append(data['img_inputs'][0][0].data[0])
    results = model(return_loss=False, rescale=True, **data)

    for corruption in cfg.corruptions:

        for severity in ['hard']:

            data_loader.dataset.pipeline.transforms[0].corruption = corruption
            # data_loader.dataset.pipeline.transforms[0].corruption = 'Clean'
            data_loader.dataset.pipeline.transforms[0].severity = severity
            data_loader_ = iter(data_loader)
            for i in range(iteration):
                data = next(data_loader_)
            imgs.append(data['img_inputs'][0][0].data[0])
            results = model(return_loss=False, rescale=True, **data)

    depth_to_image(depth_estimation, imgs)
            


if __name__ == '__main__':
    main()