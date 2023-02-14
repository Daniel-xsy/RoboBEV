import argparse
import os
import warnings
import time
import numpy as np

import torch

import mmcv
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset

from project.mmdet3d_plugin.corruptions import CORRUPTIONS


SEVERITY = {'1': 'easy', '3':'hard'}


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate Nuscenes-C Dataset')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--out', help='output result file in pickle format')
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


def save_path(corruption_root, corruption, severity, filepath):
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    # mmcv.mkdir_or_exist(os.path.join(corruption_root, corruption, SEVERITY[str(severity)], subfolder))
    return os.path.join(corruption_root, corruption, SEVERITY[str(severity)], subfolder, filename)
    

def save_multi_view_img(imgs, img_filenames, root, corruption, severity):
    """
    Args:
        img (np.array): [B, M, H, W, C]
    """
    assert imgs.shape[0] == 1, "Only support batchsize = 1"
    assert imgs.shape[1] == len(img_filenames), "Image size do not equal to filename size"
    imgs = np.squeeze(imgs)

    for i in range(len(imgs)):
        filepath = save_path(root, corruption, severity, img_filenames[i])
        mmcv.imwrite(imgs[i], filepath)
    


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
    
    # for debug purpose
    test = dataset[0]

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    print('Begin generating nuScenes-C dataset')
    for corruption in cfg.corruptions:
        print(f'Corruption type: {corruption}')  

        for severity in [1, 3]:
            print(f'\nSeverity: {severity}')  
            corrupt = CORRUPTIONS.build(dict(type=corruption, severity=severity, norm_config=cfg.img_norm_cfg))

            prog_bar = mmcv.ProgressBar(len(data_loader))
            # load_avg_time = 0.0
            # corr_avg_time = 0.0
            # save_avg_time = 0.0
            for i, data in enumerate(data_loader):
                # s = time.time()

                if i <= 1780:
                    prog_bar.update()
                    continue

                img = data['img'][0].data[0]
                img_filename = data['img_metas'][0].data[0][0]['filename']
                # e = time.time()
                # load_avg_time += (e - s)
                new_img = corrupt(img)
                # s = time.time()
                # corr_avg_time += (s - e)
                new_img = new_img.astype(np.uint8)
                
                save_multi_view_img(new_img, img_filename, cfg.corruption_root, corruption, severity)
                prog_bar.update()
                # e = time.time()
                # save_avg_time += (e - s)
                
                # print(f'load: {load_avg_time/(i+1)}\t corr: {corr_avg_time/(i+1)}\t save: {save_avg_time/(i+1)}', end='\r')
                # used for debug
                # std = cfg.img_norm_cfg['std']
                # mean = cfg.img_norm_cfg['mean']
                # orig_img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]
                # orig_img = orig_img * torch.tensor(std) + torch.tensor(mean)
                # orig_img = orig_img.numpy().astype(np.uint8)
                # mmcv.imwrite(new_img[0, 0], '/nvme/konglingdong/models/RoboDet/corruptions/new1.jpg')
                # mmcv.imwrite(orig_img[0, 0], '/nvme/konglingdong/models/RoboDet/corruptions/orig1.jpg')


if __name__ == '__main__':
    main()