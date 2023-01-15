# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
from copy import deepcopy
import os.path as osp
import pickle
import shutil
import tempfile
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info

from mmdet.core import encode_mask_results

from imagecorruptions import corrupt

import mmcv
import numpy as np
import pycocotools.mask as mask_util

from PIL import Image

norm_config = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0])

def custom_encode_mask_results(mask_results):
    """Encode bitmap mask to RLE code. Semantic Masks only
    Args:
        mask_results (list | tuple[list]): bitmap mask results.
            In mask scoring rcnn, mask_results is a tuple of (segm_results,
            segm_cls_score).
    Returns:
        list | tuple: RLE encoded mask.
    """
    cls_segms = mask_results
    num_classes = len(cls_segms)
    encoded_mask_results = []
    for i in range(len(cls_segms)):
        encoded_mask_results.append(
            mask_util.encode(
                np.array(
                    cls_segms[i][:, :, np.newaxis], order='F',
                        dtype='uint8'))[0])  # encoded with RLE
    return [encoded_mask_results]

def custom_multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    bbox_results = []
    mask_results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    have_mask = False
    for i, data in enumerate(data_loader):

        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result, dict):
                if 'bbox_results' in result.keys():
                    bbox_result = result['bbox_results']
                    batch_size = len(result['bbox_results'])
                    bbox_results.extend(bbox_result)
                if 'mask_results' in result.keys() and result['mask_results'] is not None:
                    mask_result = custom_encode_mask_results(result['mask_results'])
                    mask_results.extend(mask_result)
                    have_mask = True
            else:
                batch_size = len(result)
                bbox_results.extend(result)

            #if isinstance(result[0], tuple):
            #    assert False, 'this code is for instance segmentation, which our code will not utilize.'
            #    result = [(bbox_results, encode_mask_results(mask_results))
            #              for bbox_results, mask_results in result]
        if rank == 0:
            
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        bbox_results = collect_results_gpu(bbox_results, len(dataset))
        if have_mask:
            mask_results = collect_results_gpu(mask_results, len(dataset))
        else:
            mask_results = None
    else:
        bbox_results = collect_results_cpu(bbox_results, len(dataset), tmpdir)
        tmpdir = tmpdir+'_mask' if tmpdir is not None else None
        if have_mask:
            mask_results = collect_results_cpu(mask_results, len(dataset), tmpdir)
        else:
            mask_results = None

    if mask_results is None:
        return bbox_results
    return {'bbox_results': bbox_results, 'mask_results': mask_results}


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        '''
        bacause we change the sample of the evaluation stage to make sure that each gpu will handle continuous sample,
        '''
        #for res in zip(*part_list):
        for res in part_list:  
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    collect_results_cpu(result_part, size)


def make_corruption(img, corruption_name, norm_config):
    """make natural corruption of image
    Args:
        img (torch.Tensor): [B, M, C, H, W]
        corruption_name (str)
        norm_config (dict) 'mean', 'std
    """
    mean = norm_config['mean']
    std = norm_config['std']
    
    img = deepcopy(img)
    B, M, C, H, W = img.size()
    img = img.permute(0, 1, 3, 4, 2) # [B, M, C, H, W] => [B, M, H, W, C]
    img = img * torch.tensor(std) + torch.tensor(mean)
    # drop one camera
    img[:, 4] = 0
    # pixel value [0, 255]
    assert img.min() >= 0 and img.max() <= 255
    # new_img = torch.zeros_like(img)
    # for b in range(B):
    #     for m in range(M):
    #         # new_img[b, m] = torch.tensor(corrupt(np.uint8(img[b, m].numpy()), corruption_name=corruption_name, severity=3))
    #         new_img[b, m] = torch.tensor(iso_noise(np.uint8(img[b, m].numpy()), severity=3))

    new_img = img

    new_img = new_img - torch.tensor(mean) / torch.tensor(std)
    new_img = new_img.permute(0, 1, 4, 2, 3)

    return new_img
    

def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y


def poisson_gaussian_noise(x, severity):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
    c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    return np.uint8(x)

def low_light(x, severity):
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity]
    x = np.array(x) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity)
    return x_scaled

def iso_noise(x, severity):
    c_poisson = 25
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
    c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
    return np.uint8(x)

            

