"""
R50, Long-Term Only, No CBGS; (Phase 1)

Phase 1: We pre-train without long-term fusion for 6 epochs, similar to what is
    done in NeuralRecon. The concatenation module is simply bypassed.
Phase 2: We load Part 1 weights and continue training for for the remaining 18 
    epochs, completing the 24-epoch training cycle. 

    Changes from Phase 1 to Phase 2:
        - resume_from specified
        - do_history False -> True
        - bev_encoder_in_channels 80 -> 160
        - velocity_code_weight changed 0.2 -> 1.0
        - num_epochs 6 -> 24
"""

###############################################################################
# Training Details

_base_ = ['../_base_/datasets/nus-3d.py',
          '../_base_/default_runtime.py']

work_dir = None
load_from = None
resume_from = None
resume_optimizer = False
find_unused_parameters = False

# Because we use a custom sampler to load data in sequentially during training,
# we can only use IterBasedRunner instead of EpochBasedRunner. To train for a
# fixed # of epochs, we need to know how many iterations are in each epoch. The
# # of iters in each epoch depends on the overall batch size, which is # of 
# GPUs (num_gpus) and batch size per GPU (batch_size). "28130" is # of training
# samples in nuScenes.
num_gpus = 4
batch_size = 16
num_iters_per_epoch = 28130 // (num_gpus * batch_size)
num_epochs = 6
checkpoint_epoch_interval = 6

# Each nuScenes sequence is ~40 keyframes long. Our training procedure samples
# sequences first, then loads frames from the sampled sequence in order 
# starting from the first frame. This reduces training step-to-step diversity,
# lowering performance. To increase diversity, we split each training sequence
# in half to ~20 keyframes, and sample these shorter sequences during training.
# During testing, we do not do this splitting.
train_sequences_split_num = 2
test_sequences_split_num = 1

# By default, 3D detection datasets randomly choose another sample if there is
# no GT object in the current sample. This does not make sense when doing
# sequential sampling of frames, so we disable it.
filter_empty_gt = False

# Intermediate Checkpointing to save GPU memory.
with_cp = False

###############################################################################
# High-level Model & Training Details

base_bev_channels = 80

# Long-Term Fusion Parameters
do_history = False
history_cat_num = 16
history_cat_conv_out_channels = 160

# Short-Term Fusion Parameters
do_history_stereo_fusion = False
stereo_out_feats = 64
history_stereo_prev_step = 1
stereo_sampling_num = 7

# BEV Head Parameters
bev_encoder_in_channels = (
    base_bev_channels if not do_history else history_cat_conv_out_channels)

# Loss Weights
depth_loss_weight = 3.0
velocity_code_weight = 0.2

###############################################################################
# General Dataset & Augmentation Details.

point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]
data_config={
    'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
             'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    'Ncams': 6,
    'input_size': (256, 704),
    'src_size': (900, 1600),

    # Augmentation
    'resize': (-0.06, 0.11),
    'rot': (-5.4, 5.4),
    'flip': True,
    'crop_h': (0.0, 0.0),
    'resize_test':0.04,
}
grid_config={
    'xbound': [-51.2, 51.2, 0.8],
    'ybound': [-51.2, 51.2, 0.8],
    'zbound': [-10.0, 10.0, 20.0],
    'dbound': [2.0, 58.0, 0.5],}

voxel_size = [0.1, 0.1, 0.2] # For CenterHead

###############################################################################
# Set-up the model.

model = dict(
    type='SOLOFusion',

    # Long-Term Fusion
    do_history=do_history,
    history_cat_num=history_cat_num,
    history_cat_conv_out_channels=history_cat_conv_out_channels,

    # Short-Term Fusion
    do_history_stereo_fusion=do_history_stereo_fusion,
    history_stereo_prev_step=history_stereo_prev_step,

    # Standard R50 + FPN for Image Encoder
    img_backbone=dict(
        pretrained='torchvision://resnet50',
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=0,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=False,
        with_cp=with_cp,
        style='pytorch'),
    img_neck=dict(
        type='SECONDFPN',
        in_channels=[256, 512, 1024, 2048],
        upsample_strides=[0.25, 0.5, 1, 2],
        out_channels=[128, 128, 128, 128]),

    # A separate, smaller neck for generating stereo features. Format is
    # similar to MVS works.
    stereo_neck=None,

    # 2D -> BEV Image View Transformer.
    img_view_transformer=dict(type='ViewTransformerSOLOFusion',
                              do_history_stereo_fusion=do_history_stereo_fusion,
                              stereo_sampling_num=stereo_sampling_num,
                              loss_depth_weight=depth_loss_weight,
                              grid_config=grid_config,
                              data_config=data_config,
                              numC_Trans=base_bev_channels,
                              extra_depth_net=dict(type='ResNetForBEVDet',
                                                   numC_input=256,
                                                   num_layer=[3,],
                                                   num_channels=[256,],
                                                   stride=[1,])),
    
    # Pre-processing of BEV features before using Long-Term Fusion
    pre_process = dict(type='ResNetForBEVDet',numC_input=base_bev_channels,
                       num_layer=[2,], num_channels=[base_bev_channels,],
                       stride=[1,], backbone_output_ids=[0,]),
    
    # After using long-term fusion, process BEV for detection head.
    img_bev_encoder_backbone = dict(type='ResNetForBEVDet', 
                                    numC_input=bev_encoder_in_channels,
                                    num_channels=[base_bev_channels * 2, 
                                                  base_bev_channels * 4, 
                                                  base_bev_channels * 8],
                                    backbone_output_ids=[-1, 0, 1, 2]),
    img_bev_encoder_neck = dict(type='SECONDFPN',
                                in_channels=[bev_encoder_in_channels, 
                                             160, 320, 640],
                                upsample_strides=[1, 2, 4, 8],
                                out_channels=[64, 64, 64, 64]),
    
    # Same detection head used in BEVDet, BEVDepth, etc
    pts_bbox_head=dict(
        type='CenterHead',
        in_channels=256,
        tasks=[
            dict(num_class=1, class_names=['car']),
            dict(num_class=2, class_names=['truck', 'construction_vehicle']),
            dict(num_class=2, class_names=['bus', 'trailer']),
            dict(num_class=1, class_names=['barrier']),
            dict(num_class=2, class_names=['motorcycle', 'bicycle']),
            dict(num_class=2, class_names=['pedestrian', 'traffic_cone']),
        ],
        common_heads=dict(
            reg=(2, 2), height=(1, 2), dim=(3, 2), rot=(2, 2), vel=(2, 2)),
        share_conv_channel=64,
        bbox_coder=dict(
            type='CenterPointBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_num=500,
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            code_size=9),
        separate_head=dict(
            type='SeparateHead', init_bias=-2.19, final_kernel=3),
        loss_cls=dict(type='GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='L1Loss', reduction='mean', loss_weight=0.25),
        norm_bbox=True),
    # model training and testing settings
    train_cfg=dict(
        pts=dict(
            point_cloud_range=point_cloud_range,
            grid_size=[1024, 1024, 40],
            voxel_size=voxel_size,
            out_size_factor=8,
            dense_reg=1,
            gaussian_overlap=0.1,
            max_objs=500,
            min_radius=2,
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 
                          velocity_code_weight, velocity_code_weight])),
    test_cfg=dict(
        pts=dict(
            pc_range=point_cloud_range[:2],
            post_center_limit_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            max_per_img=500,
            max_pool_nms=False,
            min_radius=[4, 12, 10, 1, 0.85, 0.175],
            score_threshold=0.1,
            out_size_factor=8,
            voxel_size=voxel_size[:2],
            # nms_type='circle',
            pre_max_size=1000,
            post_max_size=83,
            # nms_thr=0.2,

            # Scale-NMS
            nms_type=['rotate', 'rotate', 'rotate', 'circle', 'rotate', 
                      'rotate'],
            nms_thr=[0.2, 0.2, 0.2, 0.2, 0.2, 0.5],
            nms_rescale_factor=[1.0, [0.7, 0.7], [0.4, 0.55], 1.1, [1.0, 1.0], 
                                [4.5, 9.0]]
        )))

###############################################################################
# Set-up the dataset

dataset_type = 'NuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', is_train=True, 
         data_config=data_config),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=5,
        file_client_args=file_client_args),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0],
        update_img2lidar=True),
    dict(
        type='RandomFlip3D',
        sync_2d=False,
        flip_ratio_bev_horizontal=0.5,
        flip_ratio_bev_vertical=0.5,
        update_img2lidar=True),
    dict(type='PointToMultiViewDepth', grid_config=grid_config),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['img_inputs', 'gt_bboxes_3d', 'gt_labels_3d'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='Collect3D', keys=['img_inputs'])
        ])
]
# construct a pipeline for data and gt loading in show function
# please keep its loading function consistent with test_pipeline (e.g. client)
eval_pipeline = [
    dict(type='LoadMultiViewImageFromFiles_BEVDet', data_config=data_config),
    dict(
        type='DefaultFormatBundle3D',
        class_names=class_names,
        with_label=False),
    dict(type='Collect3D', keys=['img_inputs'])
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=False)

data = dict(
    samples_per_gpu=batch_size,
    workers_per_gpu=batch_size,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        test_mode=False,
        use_valid_flag=True,
        modality=input_modality,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR',
        speed_mode=None,
        max_interval=None,
        min_interval=None,
        prev_only=None,
        fix_direction=None,
        img_info_prototype='bevdet',
        use_sequence_group_flag=True,
        sequences_split_num=train_sequences_split_num,
        filter_empty_gt=filter_empty_gt),
    val=dict(pipeline=test_pipeline, 
             classes=class_names,
             ann_file=data_root + 'nuscenes_infos_val.pkl',
             modality=input_modality, 
             img_info_prototype='bevdet',
             use_sequence_group_flag=True,
             sequences_split_num=test_sequences_split_num),
    test=dict(pipeline=test_pipeline, 
              classes=class_names,
              ann_file=data_root + 'nuscenes_infos_val.pkl',
              modality=input_modality,
              img_info_prototype='bevdet',
              use_sequence_group_flag=True,
              sequences_split_num=test_sequences_split_num))

###############################################################################
# Optimizer & Training

# Default is 2e-4 learning rate for batch size 64. When I used a smaller 
# batch size, I linearly scale down the learning rate. To do this 
# "automatically" over both per-gpu batch size and # of gpus, I set-up the
# lr as-if I'm training with batch_size per gpu for 8 GPUs below, then also
# use the autoscale-lr flag when doing training, which scales the learning
# rate based on actual # of gpus used, assuming the given learning rate is
# w.r.t 8 gpus.
lr = (2e-4 / 64) * (8 * batch_size)
optimizer = dict(type='AdamW', lr=lr, weight_decay=1e-7)

# Mixed-precision training scales the loss up by a factor before doing 
# back-propagation. I found that in early iterations, the loss, once scaled by
# 512, goes beyond the Fp16 maximum 65536 and causes nan issues. So, the 
# initial scaling here is 1.0 for "num_iters_per_epoch // 4" iters (1/4 of
# first epoch), then goes to 512.0 afterwards.
# Note that the below does not actually affect the effective loss being 
# backpropagated, it's just a trick to get FP16 to not overflow.
optimizer_config = dict(
    type='WarmupFp16OptimizerHook', 
    grad_clip=dict(max_norm=5, norm_type=2),
    warmup_loss_scale_value=1.0,
    warmup_loss_scale_iters=num_iters_per_epoch // 4,
    loss_scale=512.0
)
lr_config = None
runner = dict(
    type='IterBasedRunner', max_iters=num_epochs * num_iters_per_epoch)
checkpoint_config = dict(
    interval=checkpoint_epoch_interval * num_iters_per_epoch)
evaluation = dict(
    interval=num_epochs * num_iters_per_epoch, pipeline=eval_pipeline)
custom_hooks = [dict(
    type='ExpMomentumEMAHook', 
    resume_from=resume_from,
    resume_optimizer=resume_optimizer,
    momentum=0.001, 
    priority=49)]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])
