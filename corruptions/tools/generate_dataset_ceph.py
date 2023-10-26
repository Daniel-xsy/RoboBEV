import argparse
import os
import warnings
import time
import numpy as np
import torch
import mmcv
import cv2
from mmdet3d.datasets import build_dataset, build_dataloader
from mmcv import Config, DictAction
from mmdet3d.datasets import build_dataset
from petrel_client.client import Client
from project.mmdet3d_plugin.corruptions import CORRUPTIONS


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


def save_path(corruption_root, corruption, severity, filepath):
    """Return save path of generated corruptted images
    """
    folder, filename = os.path.split(filepath)
    _, subfolder = os.path.split(folder)
    # mmcv.mkdir_or_exist(os.path.join(corruption_root, corruption, SEVERITY[str(severity)], subfolder))
    return os.path.join(corruption_root, corruption, severity, subfolder, filename)
 
 
def save_multi_view_img_ceph(imgs, img_filenames, client, root, corruption, severity):
    """Save six view images on ceph OSS storage
    Args:
        img (np.array): [B, M, H, W, C]
    """
    assert imgs.shape[0] == 1, "Only support batchsize = 1"
    assert imgs.shape[1] == len(img_filenames), "Image size do not equal to filename size"
    imgs = np.squeeze(imgs)

    for i in range(len(imgs)):
        filepath = save_path(root, corruption, severity, img_filenames[i])
        # 图片存储
        success, img_array = cv2.imencode('.jpg', imgs[i])
        assert(success)
        img_bytes = img_array.tostring()
        client.put(filepath, img_bytes)
     

def main():

    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    print('Initialize client...')
    conf_path = '~/petreloss.conf'
    client = Client(conf_path)
    print('Initialize client over!')

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
    
    print(cfg.pretty_text)

    print('Begin generating nuScenes-C dataset')
    for corruption in cfg.corruptions:
        print(f'Corruption type: {corruption.type}')  

        for severity in ['easy', 'mid', 'hard']:
            print(f'\nSeverity: {severity}')  
            if severity not in corruption.keys():
                continue
            corrupt = CORRUPTIONS.build(dict(type=corruption.type, severity=corruption[severity], norm_config=cfg.img_norm_cfg))

            prog_bar = mmcv.ProgressBar(len(data_loader))
            for i, data in enumerate(data_loader):

                img = data['img'][0].data[0]
                img_filename = data['img_metas'][0].data[0][0]['filename']
                new_img = corrupt(img)
                new_img = new_img.astype(np.uint8)
                
                save_multi_view_img_ceph(new_img, img_filename, client, cfg.corruption_root, corruption.type, severity)
                prog_bar.update()


if __name__ == '__main__':
    main()