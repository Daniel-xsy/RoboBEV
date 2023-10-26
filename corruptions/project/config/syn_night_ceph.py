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

dataset_type = 'NuScenesDataset'
data_root = '/mnt/petrelfs/share_data/zhangjingwei/datasets/nuscenes/'
anno_root = '/mnt/petrelfs/xieshaoyuan/models/RoboBEV/data/uda/'
corruption_root = 's3://llmit/xieshaoyuan/cross_domain/'
file_client_args = dict(backend='disk')

replace_dict = {"/nvme/share/data/sets/nuScenes/": data_root,
                "/nvme/konglingdong/data/sets/nuScenes/": data_root}

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True, replace_dict=replace_dict),
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
             ann_file=anno_root + 'nuscenes_infos_daytime_val.pkl',
             pipeline=test_pipeline,
             classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler')
)

# corruptions = [dict(type='LowLight', easy=2, mid=3, hard=4)]
corruptions = [dict(type='LowLight', hard=4)]