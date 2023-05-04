_base_ = [
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/datasets/nus-mono3d.py', 
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/models/fcos3d.py',
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/schedules/mmdet_schedule_1x.py', 
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

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

file_client_args = dict(
    backend='petrel',
    path_mapping=dict({
        '/nvme/konglingdong/data/sets/nuScenes/': 's3://youquanliu/data/sets/RoboBEV/nuScenes/',
        '/nvme/share/data/sets/nuScenes/': 's3://youquanliu/data/sets/RoboBEV/nuScenes/',
    }))

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', 
         file_client_args=file_client_args),
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
    dict(type='LoadImageFromFileMono3D', 
         file_client_args=file_client_args),
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

data_root = '/nvme/konglingdong/models/RoboBEV/data/nuScenes/'
anno_root = '/nvme/konglingdong/models/RoboBEV/data/uda/'

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4,
    train=dict(
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_dry_train_mono3d.coco.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type='UDANuScenesMonoDataset',
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_dry_val_mono3d.coco.json',
        domain='dry2rain-dry',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type='UDANuScenesMonoDataset',
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_dry_val_mono3d.coco.json',
        domain='dry2rain-dry',
        img_prefix=data_root,
        pipeline=test_pipeline),)
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
evaluation = dict(interval=12)
