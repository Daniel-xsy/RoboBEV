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
corruption_root = 's3://llmit/xieshaoyuan/cross_domain/'
data_root = '/mnt/petrelfs/share_data/zhangjingwei/datasets/nuscenes/'
anno_root = '/mnt/petrelfs/xieshaoyuan/models/RoboBEV/data/uda/'

file_client_args = dict(
    backend='petrel', path_mapping={data_root: 's3://llmit/xieshaoyuan/cross_domain/',
                                        "/nvme/share/data/sets/nuScenes/": 's3://llmit/xieshaoyuan/cross_domain/',
                                        "/nvme/konglingdong/data/sets/nuScenes/": 's3://llmit/xieshaoyuan/cross_domain/'})
corruptions = ['LowLight']

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
    dict(type='Custom_LoadImageFromFileMono3D', 
         corruption_root=corruption_root,
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

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=8,
    train=dict(
        type='NuScenesMonoDataset',
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_daytime_train_mono3d.coco.json',
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset,
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_daytime_val_mono3d.coco.json',
        domain='day2night-day',
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset,
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_daytime_val_mono3d.coco.json',
        domain='day2night-day',
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
# syn daytime -> night val
# 
# mAP: 0.3045                                                                     
# mATE: 0.7651                                                                                                               
# mASE: 0.2576                                                                                                               
# mAOE: 0.5001                                                                                                               
# mAVE: 1.2102                                                                                                               
# mAAE: 0.1321                                                                                                               
# NDS: 0.3867                                                                                                                
# Eval time: 187.1s                                                                                                          
                                                                                                                           
# Per-class results:                                                                                                         
# Object Class    AP      ATE     ASE     AOE     AVE     AAE                                                                
# car     0.476   0.612   0.150   0.100   1.429   0.097                                                                      
# truck   0.209   0.821   0.198   0.143   1.418   0.187                                                                      
# bus     0.300   0.752   0.193   0.194   2.792   0.274                                                                      
# trailer 0.078   1.057   0.210   0.883   0.903   0.160                                                                      
# construction_vehicle    0.084   0.996   0.356   0.886   0.097   0.128                                                      
# pedestrian      0.399   0.706   0.299   0.727   0.972   0.132                                                              
# motorcycle      0.271   0.771   0.298   0.659   1.297   0.027                                                              
# bicycle 0.278   0.744   0.255   0.777   0.774   0.050                                                                      
# traffic_cone    0.508   0.571   0.337   nan     nan     nan                                                                
# barrier 0.442   0.619   0.279   0.132   nan     nan