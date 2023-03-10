_base_ = [
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/datasets/nus-mono3d.py', 
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/models/fcos3d.py',
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/default_runtime.py'
]
# model settings
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, False, True, True)))

class_names = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
    'motorcycle', 'pedestrian', 'traffic_cone', 'barrier'
]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

dataset_type = 'NuScenesMonoDataset'
data_root = '/nvme/share/data/sets/nuScenes/'
anno_root = '/nvme/konglingdong/models/RoboDet/data/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadAnnotations3D',
        with_bbox=True,
        with_label=True,
        with_attr_label=True,
        with_bbox_3d=True,
        with_label_3d=True,
        with_bbox_depth=True),
    dict(type='Resize', img_scale=(1600, 900), keep_ratio=True),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(
        type='Collect3D',
        keys=[
            'img', 'gt_bboxes', 'gt_labels', 'attr_labels', 'gt_bboxes_3d',
            'gt_labels_3d', 'centers2d', 'depths'
        ]),
]

test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='MultiScaleFlipAug',
        scale_factor=1.0,
        flip=False,
        transforms=[
            dict(type='RandomFlip3D'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img']),
        ])
]

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_temporal_train_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=train_pipeline,
        modality=input_modality,
        test_mode=False,
        box_type_3d='Camera'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_temporal_val_mono3d.coco.json',
        img_prefix=data_root,
        classes=class_names,
        pipeline=test_pipeline,
        modality=input_modality,
        test_mode=True,
        box_type_3d='Camera'))
# optimizer
optimizer = dict(
    lr=0.002, paramwise_cfg=dict(bias_lr_mult=2., bias_decay_mult=0.))
optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[8, 11])
total_epochs = 12
evaluation = dict(interval=2)

# mAP: 0.2979                                                                                                                                                                  
# mATE: 0.7899
# mASE: 0.2606
# mAOE: 0.4988
# mAVE: 1.2869
# mAAE: 0.1671
# NDS: 0.3773
# Eval time: 81.2s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.463   0.624   0.150   0.110   2.015   0.132
# truck   0.216   0.867   0.197   0.141   1.395   0.179
# bus     0.283   0.875   0.185   0.225   2.674   0.332
# trailer 0.098   1.151   0.229   0.706   0.465   0.113
# construction_vehicle    0.050   0.966   0.440   1.102   0.121   0.340
# pedestrian      0.398   0.708   0.287   0.726   0.897   0.154
# motorcycle      0.272   0.785   0.256   0.590   1.873   0.075
# bicycle 0.270   0.731   0.269   0.749   0.855   0.012
# traffic_cone    0.513   0.550   0.320   nan     nan     nan
# barrier 0.415   0.642   0.273   0.139   nan     nan