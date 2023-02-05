img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

plugin_dir = 'projects/mmdet3d_plugin/'

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

dataset_type = 'CustomNuScenesDataset'
data_root = '/nvme/share/data/sets/nuScenes/'
anno_root = '/nvme/konglingdong/models/RoboDet/data/'
corruption_root = '/nvme/konglingdong/data/sets/nuScenes-c/'
file_client_args = dict(backend='disk')

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    # dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=16,
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=anno_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,
             classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler')
)

corruptions = ['LowLight']
# 'MotionBlur', 'Fog', 'Snow', 'ColorQuant', 'Brightness', 'LowLight', 'CameraCrash', 'FrameLost'