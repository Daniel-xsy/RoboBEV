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
# mAP: 0.2189
# mATE: 0.8540                                                                                                                  
# mASE: 0.2758                                                                                                                  
# mAOE: 0.5361                                                                                                                  
# mAVE: 1.8479                                                                                                                  
# mAAE: 0.1279                                                                                                                  
# NDS: 0.3301                                                                                                                   
# Eval time: 34.4s                                                                                                              
                                                                                                                              
# Per-class results:                                                                                                            
# Object Class    AP      ATE     ASE     AOE     AVE     AAE                                                                   
# car     0.402   0.682   0.151   0.144   1.318   0.165                                                                         
# truck   0.147   0.897   0.207   0.166   1.577   0.200                                                                         
# bus     0.118   0.922   0.197   0.337   4.127   0.082
# trailer 0.026   1.183   0.272   1.034   2.666   0.050
# construction_vehicle    0.004   1.120   0.424   1.083   0.133   0.295
# pedestrian      0.219   0.820   0.326   0.873   0.973   0.181
# motorcycle      0.256   0.922   0.271   0.536   1.942   0.000
# bicycle 0.174   0.688   0.285   0.554   2.047   0.049
# traffic_cone    0.429   0.663   0.331   nan     nan     nan
# barrier 0.413   0.644   0.293   0.099   nan     nan