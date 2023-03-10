_base_ = './fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py'
# model settings
model = dict(
    train_cfg=dict(
        code_weight=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.05, 0.05]))
# optimizer
optimizer = dict(lr=0.001)
load_from = 'work_dirs/fcos3d_nus/latest.pth'

# Evaluating bboxes of img_bbox
# mAP: 0.3214                                                                                                                                                                
# mATE: 0.7538
# mASE: 0.2603
# mAOE: 0.4864
# mAVE: 1.3321
# mAAE: 0.1574
# NDS: 0.3949
# Eval time: 73.1s

# Per-class results:
# Object Class    AP      ATE     ASE     AOE     AVE     AAE
# car     0.487   0.591   0.150   0.110   1.997   0.130
# truck   0.235   0.847   0.205   0.149   1.349   0.198
# bus     0.313   0.798   0.182   0.157   3.001   0.280
# trailer 0.115   1.149   0.223   0.751   0.546   0.074
# construction_vehicle    0.057   0.995   0.439   1.126   0.132   0.290
# pedestrian      0.412   0.684   0.286   0.711   0.886   0.161
# motorcycle      0.298   0.722   0.263   0.562   1.915   0.120
# bicycle 0.300   0.677   0.264   0.668   0.831   0.006
# traffic_cone    0.541   0.503   0.316   nan     nan     nan
# barrier 0.456   0.572   0.275   0.142   nan     nan