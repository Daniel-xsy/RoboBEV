import argparse
import os
import warnings
import time
import numpy as np
import torch
import mmcv
import cv2 as cv
import matplotlib.pylab as plt

from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
from project.mmdet3d_plugin.corruptions import CORRUPTIONS


CAMERA = {'CAM_FRONT_LEFT':2, 
          'CAM_FRONT': 0, 
          'CAM_FRONT_RIGHT': 1,
          'CAM_BACK_LEFT':4, 
          'CAM_BACK':3, 
          'CAM_BACK_RIGHT':5}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Nuscenes-C Dataset')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    args = parser.parse_args()

    return args


def plot_histogram(img):
    '''Plot histogram of given image
    Args:
        img (numpy.array): [H, W, C]
    '''


def plot_subfig(imgs, subplot, filename, titles=None, figsize=(10,10), histogram=False):
    '''Plot subplot of given images
    Args:
        img (List): n x [H, W, C]
        subplot (List): [H ,W]
    '''
    assert len(subplot) == 2, 'Please provide subplot width and length'
    height, width = subplot
    assert len(imgs) == height * width, f'Subplot and image number mismatch: subplot {subplot}, img num {len(imgs)}'
    if titles is not None:
        assert len(imgs) == len(titles), 'Image number shoul equal to titles.'

    plt.figure(figsize=(2 * width, 2 * height), dpi=300)

    for h in range(height):
        for w in range(width):
            # Plot image
            if not histogram:
                ax = plt.subplot(height, width, h * width + w + 1)
                plt.axis('off')
                plt.imshow(imgs[h * width + w])
                # plt.tight_layout()
                if titles is not None:
                    ax.set_title(titles[h * width + w], fontdict={'fontsize':6})
            # Plot histogram
            else:
                ax = plt.subplot(height, width, h * width + w + 1)
                plt.hist(np.ravel(imgs[h * width + w]), 20, [0,256])
                ax.set_ylim(0, 1e6)
                # Turn off yticks of other images
                if w != 0:
                    plt.yticks([])
                if titles is not None:
                    ax.set_title(titles[h * width + w])
    if not histogram:
        plt.subplots_adjust(left=0, bottom=0.2, right=1, top=0.85, wspace=0.01, hspace=0.05)
    # plt.tight_layout()
    plt.savefig(filename)


def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, 'plugin'):
        if cfg.plugin:
            import importlib
            if hasattr(cfg, 'plugin_dir'):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split('/')
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + '.' + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # validation set
    cfg.data.val.test_mode = True
    dataset = build_dataset(cfg.data.val)

    imgs = []

    for corruption in cfg.corruptions:
        print(f'Corruption type: {corruption}')  
        if corruption == 'Clean':
            dataset.pipeline.transforms[0].corruption = corruption
            severity = None
            for sample_id in cfg.sample_id:
                img = dataset[sample_id]['img'][0].data[CAMERA[cfg.camera_id]]
                img = img.permute(1, 2, 0).numpy().astype(np.uint8)
                imgs.append(img)
                filename = os.path.join(cfg.save_path, 'clean', f'{cfg.sample_id}_{cfg.camera_id}.png')
                mmcv.imwrite(img, filename)
        else:
            for severity in cfg.severity:
                dataset.pipeline.transforms[0].corruption = corruption
                dataset.pipeline.transforms[0].severity = severity
                for sample_id in cfg.sample_id:
                    img = dataset[sample_id]['img'][0].data[CAMERA[cfg.camera_id]]
                    img = img.permute(1, 2, 0).numpy().astype(np.uint8)
                    imgs.append(img)
                    filename = os.path.join(cfg.save_path, corruption, severity, f'{cfg.sample_id}_{cfg.camera_id}.png')
                    mmcv.imwrite(img, filename)
            
    if cfg.subplot is not None:
        plot_subfig(imgs, cfg.subplot, cfg.subplot_path, cfg.titles, histogram=cfg.histogram)

            
if __name__ == '__main__':
    main()