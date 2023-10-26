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
        ann_file=anno_root + 'nuscenes_infos_rain_val_mono3d.coco.json',
        domain='dry2rain-rain',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset,
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_rain_val_mono3d.coco.json',
        domain='dry2rain-rain',
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
# rain val dataset
#
# mAP: 0.2063
# mATE: 0.8600                                                                                                              
# mASE: 0.2704                                                                                                              
# mAOE: 0.5810                                                                                                              
# mAVE: 2.0521                                                                                                              
# mAAE: 0.1498                                                                                                              
# NDS: 0.3170                                                                                                               
# Eval time: 34.6s                                                                                                          
                                                                                                                          
# Per-class results:                                                                                                        
# Object Class    AP      ATE     ASE     AOE     AVE     AAE                                                               
# car     0.413   0.662   0.148   0.134   1.365   0.178                                                                     
# truck   0.134   0.909   0.207   0.180   1.505   0.176                                                                     
# bus     0.097   0.844   0.191   0.320   3.886   0.272                                                                     
# trailer 0.037   1.130   0.289   1.008   2.033   0.075                                                                     
# construction_vehicle    0.012   1.283   0.402   0.918   0.114   0.242                                                     
# pedestrian      0.220   0.845   0.317   0.900   0.974   0.192                                                             
# motorcycle      0.077   0.914   0.229   0.952   4.639   0.022                                                             
# bicycle 0.195   0.726   0.288   0.704   1.901   0.041                                                                     
# traffic_cone    0.424   0.685   0.344   nan     nan     nan                                                               
# barrier 0.454   0.600   0.291   0.114   nan     nan