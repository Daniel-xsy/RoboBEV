_base_ = [
    './beverse_singleframe_small.py'
]

receptive_field = 3
future_frames = 4
future_discount = 0.95

model = dict(
    temporal_model=dict(
        type='Temporal3DConvModel',
        receptive_field=receptive_field,
        input_egopose=True,
        in_channels=64,
        input_shape=(128, 128),
        with_skip_connect=True,
    ),
    pts_bbox_head=dict(
        task_enbale={
            '3dod': True, 'map': True, 'motion': True,
        },
        task_weights={
            '3dod': 1.0, 'map': 10.0, 'motion': 1.0,
        },
        cfg_motion=dict(
            type='IterativeFlow',
            task_dict={
                'segmentation': 2,
                'instance_center': 1,
                'instance_offset': 2,
                'instance_flow': 2,
            },
            receptive_field=receptive_field,
            n_future=future_frames,
            using_spatial_prob=True,
            n_gru_blocks=1,
            future_discount=future_discount,
            loss_weights={
                'loss_motion_seg': 5.0,
                'loss_motion_centerness': 1.0,
                'loss_motion_offset': 1.0,
                'loss_motion_flow': 1.0,
                'loss_motion_prob': 10.0,
            },
            sample_ignore_mode='past_valid',
            posterior_with_label=False,
        ),
    ),
    train_cfg=dict(
        pts=dict(
            code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ),
    ),
)

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        dataset=dict(
            receptive_field=receptive_field,
            future_frames=future_frames,
        ),
    ),
    val=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
    test=dict(
        receptive_field=receptive_field,
        future_frames=future_frames,
    ),
)
corruptions = ['ColorQuant', 'Brightness', 'LowLight', 'CameraCrash', 'FrameLost']

# mAP: 0.3512                                                                                                                                
# mATE: 0.6243
# mASE: 0.2694
# mAOE: 0.3999
# mAVE: 0.3292
# mAAE: 0.1827
# NDS: 0.4951
# Eval time: 83.0s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.562   0.456   0.154   0.073   0.311   0.188
# truck   0.264   0.618   0.209   0.085   0.289   0.200
# bus     0.365   0.673   0.193   0.068   0.780   0.337
# trailer 0.136   0.994   0.240   0.487   0.205   0.038
# construction_vehicle    0.072   0.897   0.485   1.196   0.106   0.327
# pedestrian      0.418   0.629   0.296   0.637   0.409   0.172
# motorcycle      0.344   0.605   0.259   0.376   0.414   0.197
# bicycle 0.301   0.487   0.253   0.541   0.121   0.002
# traffic_cone    0.533   0.417   0.330   nan     nan     nan
# barrier 0.515   0.467   0.276   0.137   nan     nan