import mmcv
import os
import sys
from mmcv.runner import (get_dist_info)


def collect_metric(results, logging):

    assert type(results) == dict, f'Results should be metric dict, but now {type(results)}'
    if 'pts_bbox_NuScenes/NDS' in results.keys():
        prefix = 'pts'
    elif 'img_bbox_NuScenes/NDS' in results.keys():
        prefix = 'img'
    else:
        raise KeyError
    logging.write('Evaluating Results\n')
    logging.write('| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |')
    logging.write('| ------- | ------- | -------- | -------- | -------- | -------- | -------- |')
    NDS = results[f'{prefix}_bbox_NuScenes/NDS']
    mAP = results[f'{prefix}_bbox_NuScenes/mAP']
    mATE = results[f'{prefix}_bbox_NuScenes/mATE']
    mASE = results[f'{prefix}_bbox_NuScenes/mASE']
    mAOE = results[f'{prefix}_bbox_NuScenes/mAOE']
    mAVE = results[f'{prefix}_bbox_NuScenes/mAVE']
    mAAE = results[f'{prefix}_bbox_NuScenes/mAAE']
    logging.write(f'| {NDS:.4f}    | {mAP:.4f}    | {mATE:.4f}     | {mASE:.4f}     | {mAOE:.4f}     | {mAVE:.4f}     | {mAAE:.4f}     |\n')


def collect_average_metric(results_list, logging):

    assert type(results_list) == list or tuple, f'Results should be list of metric, but now {type(results_list)}'
    if 'pts_bbox_NuScenes/NDS' in results_list[0].keys():
        prefix = 'pts'
    elif 'img_bbox_NuScenes/NDS' in results_list[0].keys():
        prefix = 'img'
    else:
        raise KeyError
    NDS = 0
    mAP = 0
    mATE = 0
    mASE = 0
    mAOE = 0
    mAVE = 0
    mAAE = 0

    for results in results_list:
        NDS += results[f'{prefix}_bbox_NuScenes/NDS']
        mAP += results[f'{prefix}_NuScenes/mAP']
        mATE += results[f'{prefix}_NuScenes/mATE']
        mASE += results[f'{prefix}_NuScenes/mASE']
        mAOE += results[f'{prefix}_NuScenes/mAOE']
        mAVE += results[f'{prefix}_NuScenes/mAVE']
        mAAE += results[f'{prefix}_NuScenes/mAAE']

    NDS /= len(results_list)
    mAP /= len(results_list)
    mATE /= len(results_list)
    mASE /= len(results_list)
    mAOE /= len(results_list)
    mAVE /= len(results_list)
    mAAE /= len(results_list)

    logging.write('| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |')
    logging.write('| ------- | ------- | -------- | -------- | -------- | -------- | -------- |')
    logging.write(f'| {NDS:.4f}    | {mAP:.4f}    | {mATE:.4f}     | {mASE:.4f}     | {mAOE:.4f}     | {mAVE:.4f}     | {mAAE:.4f}     |\n')


class Logging_str(object):
    def __init__(self, logfile_path):
        self.logfile_path = logfile_path

    def write(self, str = None):
        assert str is not None
        rank, _ = get_dist_info()
        if rank == 0:
            with open(self.logfile_path, "a") as file_object:
                msg = str
                file_object.write(msg+'\n')
            print(str)


if __name__=='__main__':
    filepath = 'results_dict_list.pkl'
    results_dict_list = mmcv.load(filepath)
    logging = Logging_str('test.md')
    collect_metric(results_dict_list[0], logging)
    collect_average_metric(results_dict_list, logging)