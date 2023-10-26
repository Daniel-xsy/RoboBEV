_base_ = [
    '/mnt/petrelfs/xieshaoyuan/models/mmdetection3d/configs/_base_/datasets/nus-mono3d.py', 
    '/mnt/petrelfs/xieshaoyuan/models/mmdetection3d/configs/_base_/models/fcos3d.py',
    '/mnt/petrelfs/xieshaoyuan/models/mmdetection3d/configs/_base_/schedules/mmdet_schedule_1x.py', 
    '/mnt/petrelfs/xieshaoyuan/models/mmdetection3d/configs/_base_/default_runtime.py'
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

dataset = 'UDANuScenesMonoDataset'
data_root = '/mnt/petrelfs/share_data/zhangjingwei/datasets/nuscenes/'
anno_root = '/mnt/petrelfs/xieshaoyuan/models/RoboBEV/data/uda/'

replace_dict = {"/nvme/share/data/sets/nuScenes/": data_root,
                "/nvme/konglingdong/data/sets/nuScenes/": data_root}

corruption = {'LowLight': 0.1,
              'Fog': 0.5,
              'Snow': 0.4}
severity = {'easy': 0.9, 'mid': 0.1}

train_pipeline = [
    dict(type='LoadImageFromFileMono3D', replace_dict=replace_dict),
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
    dict(type='CorruptAugmentMono', p=0.2, corruption=corruption, severity=severity),
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
    dict(type='LoadImageFromFileMono3D', replace_dict=replace_dict),
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
    workers_per_gpu=8,
    train=dict(
        type='NuScenesMonoDataset',
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_dry_train_mono3d.coco.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset,
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_dry_val_mono3d.coco.json',
        domain='dry2rain-dry',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset,
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
evaluation = dict(interval=2)

#
# dry val dataset
#
# mAP: 0.2961                                                                
# mATE: 0.7669                                                                                                              
# mASE: 0.2547                                                                                                              
# mAOE: 0.4775                                                                                                              
# mAVE: 1.2976                                                                                                              
# mAAE: 0.1367                                                                                                              
# NDS: 0.3844                                                                                                               
# Eval time: 148.4s                                                                                                         
                                                                                                                          
# Per-class results:                                                                                                        
# Object Class    AP      ATE     ASE     AOE     AVE     AAE                                                               
# car     0.462   0.627   0.148   0.107   1.887   0.108                                                                     
# truck   0.178   0.815   0.199   0.175   1.324   0.270                                                                     
# bus     0.197   0.902   0.189   0.233   1.970   0.338                                                                     
# trailer 0.083   1.034   0.217   0.877   1.159   0.074                                                                     
# construction_vehicle    0.121   1.062   0.365   0.656   0.094   0.126                                                     
# pedestrian      0.388   0.709   0.300   0.725   0.907   0.143                                                             
# motorcycle      0.316   0.683   0.264   0.640   1.724   0.015                                                             
# bicycle 0.239   0.731   0.270   0.748   1.316   0.019                                                                     
# traffic_cone    0.522   0.529   0.301   nan     nan     nan                                                               
# barrier 0.453   0.578   0.293   0.135   nan     nan