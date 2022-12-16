import argparse
import os

import onnx
import tensorrt as trt
import torch
import torch.onnx
from mmcv import Config
from mmdeploy.backend.tensorrt.utils import from_onnx

try:
    # If mmdet version > 2.23.0, compat_cfg would be imported and
    # used from mmdet instead of mmdet3d.
    from mmdet.utils import compat_cfg
except ImportError:
    from mmdet3d.utils import compat_cfg

from mmcv.runner import load_checkpoint
from mmdeploy.backend.tensorrt import load_tensorrt_plugin

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet.datasets import replace_ImageToTensor
from tools.misc.fuse_conv_bn import fuse_module


def parse_args():
    parser = argparse.ArgumentParser(description='Deploy BEVDet with Tensorrt')
    parser.add_argument('config', help='deploy config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('work_dir', help='work dir to save file')
    parser.add_argument(
        '--prefix', default='bevdet', help='prefix of the save file name')
    parser.add_argument(
        '--fp16', action='store_true', help='Whether to use tensorrt fp16')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    args = parser.parse_args()
    return args


def get_plugin_names():
    return [pc.name for pc in trt.get_plugin_registry().plugin_creator_list]


def main():
    args = parse_args()
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)

    load_tensorrt_plugin()
    assert 'bev_pool_v2' in get_plugin_names(), \
        'bev_pool_v2 is not in the plugin list of tensorrt, ' \
        'please install mmdeploy from ' \
        'https://github.com/HuangJunJie2017/mmdeploy.git'

    model_prefix = args.prefix
    if args.fp16:
        model_prefix = model_prefix + '_fp16'

    cfg = Config.fromfile(args.config)
    cfg.model.pretrained = None
    cfg.model.type = cfg.model.type + 'TRT'

    cfg = compat_cfg(cfg)
    cfg.gpu_ids = [0]

    # build the dataloader
    test_dataloader_default_args = dict(
        samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False)

    if isinstance(cfg.data.test, dict):
        cfg.data.test.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.test.pipeline = replace_ImageToTensor(
                cfg.data.test.pipeline)
    elif isinstance(cfg.data.test, list):
        for ds_cfg in cfg.data.test:
            ds_cfg.test_mode = True
        if cfg.data.test_dataloader.get('samples_per_gpu', 1) > 1:
            for ds_cfg in cfg.data.test:
                ds_cfg.pipeline = replace_ImageToTensor(ds_cfg.pipeline)

    test_loader_cfg = {
        **test_dataloader_default_args,
        **cfg.data.get('test_dataloader', {})
    }
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    assert model.img_view_transformer.grid_size[0] == 128
    assert model.img_view_transformer.grid_size[1] == 128
    assert model.img_view_transformer.grid_size[2] == 1
    load_checkpoint(model, args.checkpoint, map_location='cpu')
    if args.fuse_conv_bn:
        model_prefix = model_prefix + '_fuse'
        model = fuse_module(model)
    model.cuda()
    model.eval()

    for i, data in enumerate(data_loader):
        inputs = [t.cuda() for t in data['img_inputs'][0]]
        metas = model.get_bev_pool_input(inputs)
        img = inputs[0].squeeze(0)
        with torch.no_grad():
            torch.onnx.export(
                model,
                (img.float().contiguous(),
                 metas[1].int().contiguous(),
                 metas[2].int().contiguous(),
                 metas[0].int().contiguous(),
                 metas[3].int().contiguous(),
                 metas[4].int().contiguous()),
                args.work_dir + model_prefix + '.onnx',
                opset_version=11,
                input_names=[
                    'img', 'ranks_depth', 'ranks_feat', 'ranks_bev',
                    'interval_starts', 'interval_lengths'
                ],
                output_names=[f'output_{j}' for j in range(36)])
        break
    # check onnx model
    onnx_model = onnx.load(args.work_dir + model_prefix + '.onnx')
    try:
        onnx.checker.check_model(onnx_model)
    except Exception:
        print('ONNX Model Incorrect')
    else:
        print('ONNX Model Correct')

    # convert to tensorrt
    num_points = metas[0].shape[0]
    num_intervals = metas[3].shape[0]
    img_shape = img.shape
    from_onnx(
        args.work_dir + model_prefix + '.onnx',
        args.work_dir + model_prefix,
        fp16_mode=args.fp16,
        max_workspace_size=1 << 30,
        input_shapes=dict(
            img=dict(
                min_shape=img_shape, opt_shape=img_shape, max_shape=img_shape),
            ranks_depth=dict(
                min_shape=[num_points],
                opt_shape=[num_points],
                max_shape=[num_points]),
            ranks_feat=dict(
                min_shape=[num_points],
                opt_shape=[num_points],
                max_shape=[num_points]),
            ranks_bev=dict(
                min_shape=[num_points],
                opt_shape=[num_points],
                max_shape=[num_points]),
            interval_starts=dict(
                min_shape=[num_intervals],
                opt_shape=[num_intervals],
                max_shape=[num_intervals]),
            interval_lengths=dict(
                min_shape=[num_intervals],
                opt_shape=[num_intervals],
                max_shape=[num_intervals])))


if __name__ == '__main__':

    main()
