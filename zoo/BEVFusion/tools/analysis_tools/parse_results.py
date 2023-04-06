import mmcv
import os
import sys
import numpy as np
from mmcv.runner import (get_dist_info)


def collect_metric(results, logging):

    assert type(results) == dict, f'Results should be metric dict, but now {type(results)}'

    logging.write('Evaluating Results\n')
    logging.write('| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |')
    logging.write('| ------- | ------- | -------- | -------- | -------- | -------- | -------- |')
    NDS = results['object/nds']
    mAP = results['object/map']
    mATE = results['object/mATE']
    mASE = results['object/mASE']
    mAOE = results['object/mAOE']
    mAVE = results['object/mAVE']
    mAAE = results['object/mAAE']
    logging.write(f'| {NDS:.4f}    | {mAP:.4f}    | {mATE:.4f}     | {mASE:.4f}     | {mAOE:.4f}     | {mAVE:.4f}     | {mAAE:.4f}     |\n')


def collect_average_metric(results_list, logging):

    assert type(results_list) == list or tuple, f'Results should be list of metric, but now {type(results_list)}'

    NDS = 0
    mAP = 0
    mATE = 0
    mASE = 0
    mAOE = 0
    mAVE = 0
    mAAE = 0

    for results in results_list:
        NDS += results['object/nds']
        mAP += results['object/map']
        mATE += results['object/mATE']
        mASE += results['object/mASE']
        mAOE += results['object/mAOE']
        mAVE += results['object/mAVE']
        mAAE += results['object/mAAE']

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

############################# Debug #############################
# if __name__=='__main__':
#     results = {'object/car_ap_dist_0.5': 0.168, 'object/car_ap_dist_1.0': 0.4667, 'object/car_ap_dist_2.0': 0.7078, 'object/car_ap_dist_4.0': 0.809, 'object/car_trans_err': 0.5293, 'object/car_scale_err': 0.156, 'object/car_orient_err': 0.1278, 'object/car_vel_err': 0.981, 'object/car_attr_err': 0.223, 'object/mATE': 0.6677, 'object/mASE': 0.2727, 'object/mAOE': 0.5612, 'object/mAVE': 0.8954, 'object/mAAE': 0.2593, 'object/truck_ap_dist_0.5': 0.0278, 'object/truck_ap_dist_1.0': 0.1761, 'object/truck_ap_dist_2.0': 0.3998, 'object/truck_ap_dist_4.0': 0.5583, 'object/truck_trans_err': 0.723, 'object/truck_scale_err': 0.2154, 'object/truck_orient_err': 0.1434, 'object/truck_vel_err': 0.9053, 'object/truck_attr_err': 0.231, 'object/construction_vehicle_ap_dist_0.5': 0.0, 'object/construction_vehicle_ap_dist_1.0': 0.0335, 'object/construction_vehicle_ap_dist_2.0': 0.1319, 'object/construction_vehicle_ap_dist_4.0': 0.2041, 'object/construction_vehicle_trans_err': 0.8889, 'object/construction_vehicle_scale_err': 0.5052, 'object/construction_vehicle_orient_err': 1.097, 'object/construction_vehicle_vel_err': 0.1517, 'object/construction_vehicle_attr_err': 0.3834, 'object/bus_ap_dist_0.5': 0.0419, 'object/bus_ap_dist_1.0': 0.26, 'object/bus_ap_dist_2.0': 0.5777, 'object/bus_ap_dist_4.0': 0.7348, 'object/bus_trans_err': 0.7138, 'object/bus_scale_err': 0.1875, 'object/bus_orient_err': 0.0906, 'object/bus_vel_err': 1.557, 'object/bus_attr_err': 0.3072, 'object/trailer_ap_dist_0.5': 0.0, 'object/trailer_ap_dist_1.0': 0.0542, 'object/trailer_ap_dist_2.0': 0.2596, 'object/trailer_ap_dist_4.0': 0.4009, 'object/trailer_trans_err': 0.9614, 'object/trailer_scale_err': 0.238, 'object/trailer_orient_err': 0.4951, 'object/trailer_vel_err': 0.7632, 'object/trailer_attr_err': 0.0735, 'object/barrier_ap_dist_0.5': 0.1993, 'object/barrier_ap_dist_1.0': 0.5323, 'object/barrier_ap_dist_2.0': 0.6456, 'object/barrier_ap_dist_4.0': 0.6924, 'object/barrier_trans_err': 0.4887, 'object/barrier_scale_err': 0.2853, 'object/barrier_orient_err': 0.1365, 'object/barrier_vel_err': np.nan, 'object/barrier_attr_err': np.nan, 'object/motorcycle_ap_dist_0.5': 0.105, 'object/motorcycle_ap_dist_1.0': 0.2767, 'object/motorcycle_ap_dist_2.0': 0.461, 'object/motorcycle_ap_dist_4.0': 0.5219, 'object/motorcycle_trans_err': 0.6465, 'object/motorcycle_scale_err': 0.2468, 'object/motorcycle_orient_err': 0.5295, 'object/motorcycle_vel_err': 1.5659, 'object/motorcycle_attr_err': 0.0878, 'object/bicycle_ap_dist_0.5': 0.1024, 'object/bicycle_ap_dist_1.0': 0.2326, 'object/bicycle_ap_dist_2.0': 0.34, 'object/bicycle_ap_dist_4.0': 0.3825, 'object/bicycle_trans_err': 0.5577, 'object/bicycle_scale_err': 0.2567, 'object/bicycle_orient_err': 1.0526, 'object/bicycle_vel_err': 0.3786, 'object/bicycle_attr_err': 0.0184, 'object/pedestrian_ap_dist_0.5': 0.0747, 'object/pedestrian_ap_dist_1.0': 0.3064, 'object/pedestrian_ap_dist_2.0': 0.5155, 'object/pedestrian_ap_dist_4.0': 0.6462, 'object/pedestrian_trans_err': 0.7035, 'object/pedestrian_scale_err': 0.3008, 'object/pedestrian_orient_err': 1.3782, 'object/pedestrian_vel_err': 0.8608, 'object/pedestrian_attr_err': 0.7504, 'object/traffic_cone_ap_dist_0.5': 0.2524, 'object/traffic_cone_ap_dist_1.0': 0.5279, 'object/traffic_cone_ap_dist_2.0': 0.6678, 'object/traffic_cone_ap_dist_4.0': 0.729, 'object/traffic_cone_trans_err': 0.4641, 'object/traffic_cone_scale_err': 0.3349, 'object/traffic_cone_orient_err': np.nan, 'object/traffic_cone_vel_err': np.nan, 'object/traffic_cone_attr_err': np.nan, 'object/nds': 0.4121622151683061, 'object/map': 0.35559372622755425}
#     mATE = results['object/mATE']
#     mASE = results['object/mASE']
#     mAOE = results['object/mAOE']
#     mAVE = results['object/mAVE']
#     mAAE = results['object/mAAE']

# mAP: 0.3556                              
# mATE: 0.6677
# mASE: 0.2727
# mAOE: 0.5612
# mAVE: 0.8954
# mAAE: 0.2593
# NDS: 0.4122

