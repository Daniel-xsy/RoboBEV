_base_ = [
    './beverse_singleframe_tiny.py'
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
    samples_per_gpu=2,
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
# 'MotionBlur', 'Fog', 'Snow', 


# Evaluating bboxes of pts_bbox
# mAP: 0.3214                                                                                                                               
# mATE: 0.6807
# mASE: 0.2782
# mAOE: 0.4657
# mAVE: 0.3281
# mAAE: 0.1893
# NDS: 0.4665
# Eval time: 99.2s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.534   0.508   0.162   0.089   0.321   0.194
# truck   0.236   0.659   0.212   0.094   0.277   0.210
# bus     0.312   0.719   0.202   0.071   0.694   0.317
# trailer 0.130   1.050   0.229   0.527   0.222   0.044
# construction_vehicle    0.064   0.986   0.492   1.097   0.115   0.366
# pedestrian      0.382   0.708   0.299   0.728   0.459   0.218
# motorcycle      0.299   0.710   0.261   0.649   0.408   0.163
# bicycle 0.252   0.486   0.298   0.805   0.129   0.004
# traffic_cone    0.514   0.481   0.344   nan     nan     nan
# barrier 0.491   0.500   0.281   0.132   nan     nan