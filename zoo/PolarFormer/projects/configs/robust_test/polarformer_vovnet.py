_base_ = [
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/datasets/nus-3d.py',
    '/nvme/konglingdong/models/mmdetection3d/configs/_base_/default_runtime.py'
]

plugin=True
plugin_dir='projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
radius_range=[1., 65., 1.] # [start, end, interval]
grid_res = 0.8 
voxel_size = [grid_res, grid_res, grid_res]

output_size = [256, 64, 10] # [azimuth, radius, height]
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[57.375, 57.120, 58.395], to_rgb=False) # different from r101
# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

model = dict(
    type='PolarFormer',
    use_grid_mask=True,
    img_backbone=dict(
        type='VoVNet',
        spec_name='V-99-eSE',
        norm_eval=True,
        frozen_stages=1,
        input_ch=3,
        out_features=['stage3', 'stage4', 'stage5']),
    img_neck=dict(
        type='FPN_TRANS',
        num_encoder=0, # encoder is not used here
        num_decoder=3,
        num_levels=3,
        radius_range=radius_range,
        use_different_res=True,
        use_bev_aug=True,
        output_multi_scale=True,
        grid_res=grid_res,
        pc_range=point_cloud_range,
        output_size=output_size,
        fpn_cfg=dict(
                in_channels=[512, 768, 1024],
                out_channels=256,
                start_level=1,
                add_extra_convs='on_output',
                num_outs=3,
                relu_before_extra_convs=True),
            ),
    pts_bbox_head=dict(
        type='PolarFormerHead',
        num_query=900,
        num_classes=10,
        in_channels=256,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        radius_range=radius_range,
        code_weights = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2],
        transformer=dict(
            type='PolarTransformer',
            num_feature_levels=3,
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=6,
                transformerlayers=dict(
                    type='BaseTransformerLayer',
                    attn_cfgs=dict(
                        type='MultiScaleDeformableAttention', embed_dims=256, num_levels=3),
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'ffn', 'norm'))),
             decoder=dict(
                type='PolarTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                pc_range=point_cloud_range,
                radius_range=radius_range,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=256,
                            num_heads=8,
                            dropout=0.1),
                        dict(
                            type='MultiScaleDeformableAttention',
                            embed_dims=256,
                            num_levels=3)
                    ],
                    feedforward_channels=1024,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10), 
        positional_encoding=dict(
            type='SinePositionalEncoding',
            num_feats=128,
            normalize=True,
            offset=-0.5),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head. 
            pc_range=point_cloud_range))))

dataset_type = 'TransNuScenesDataset'
data_root = '/nvme/share/data/sets/nuScenes/'
anno_root = '../../data/'

file_client_args = dict(backend='disk')

db_sampler = dict(
    data_root=data_root,
    info_path=data_root + 'nuscenes_dbinfos_train.pkl',
    rate=1.0,
    prepare=dict(
        filter_by_difficulty=[-1],
        filter_by_min_points=dict(
            car=5,
            truck=5,
            bus=5,
            trailer=5,
            construction_vehicle=5,
            traffic_cone=5,
            barrier=5,
            motorcycle=5,
            bicycle=5,
            pedestrian=5)),
    classes=class_names,
    sample_groups=dict(
        car=2,
        truck=3,
        construction_vehicle=7,
        bus=4,
        trailer=6,
        barrier=2,
        motorcycle=6,
        bicycle=6,
        pedestrian=2,
        traffic_cone=2),
    points_loader=dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=5,
        use_dim=[0, 1, 2, 3, 4],
        file_client_args=file_client_args))

train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='Collect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'],
                           meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 
                            'cam2lidar', 'cam_intrinsic',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                            'img_norm_cfg', 'pcd_trans', 'sample_idx',
                            'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                            'transformation_3d_flow'))
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
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
            dict(type='Collect3D', keys=['img'],
                                   meta_keys=('filename', 'ori_shape', 'img_shape', 'lidar2img', 
                                    'cam2lidar', 'cam_intrinsic',
                                    'depth2img', 'cam2img', 'pad_shape',
                                    'scale_factor', 'flip', 'pcd_horizontal_flip',
                                    'pcd_vertical_flip', 'box_mode_3d', 'box_type_3d',
                                    'img_norm_cfg', 'pcd_trans', 'sample_idx',
                                    'pcd_scale_factor', 'pcd_rotation', 'pts_filename',
                                    'transformation_3d_flow'))
        ])
]


data = dict(
    samples_per_gpu=1,
    workers_per_gpu=6,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(
        type=dataset_type, 
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline, 
        classes=class_names, 
        modality=input_modality),
    test=dict(
        type=dataset_type, 
        data_root=data_root,
        ann_file=anno_root + 'nuscenes_infos_temporal_val.pkl',
        pipeline=test_pipeline, 
        classes=class_names, 
        modality=input_modality))

optimizer = dict(
    type='AdamW', 
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1),
        }),
    weight_decay=0.075)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 24
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)
load_from='pretrained/dd3_det_final.pth'
find_unused_parameters=True


# mAP: 0.5004                                                                                                                                                                       
# mATE: 0.5826
# mASE: 0.2621
# mAOE: 0.2473
# mAVE: 0.6015
# mAAE: 0.1926
# NDS: 0.5616
# Eval time: 116.7s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.687   0.393   0.147   0.046   0.651   0.203
# truck   0.469   0.617   0.207   0.059   0.604   0.225
# bus     0.580   0.569   0.196   0.048   1.107   0.250
# trailer 0.325   0.849   0.221   0.206   0.438   0.151
# construction_vehicle    0.193   0.947   0.447   0.508   0.145   0.360
# pedestrian      0.554   0.549   0.293   0.452   0.406   0.193
# motorcycle      0.493   0.555   0.240   0.292   1.089   0.152
# bicycle 0.453   0.496   0.271   0.492   0.371   0.006
# traffic_cone    0.637   0.390   0.309   nan     nan     nan
# barrier 0.613   0.461   0.290   0.123   nan     nan