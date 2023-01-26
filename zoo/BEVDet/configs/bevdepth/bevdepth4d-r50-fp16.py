# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['./bevdepth4d-r50.py']

# avoid the type error of the bias of DCNv2
model = dict(
    img_view_transformer=dict(dcn_config=dict(bias=False)))

fp16 = dict(loss_scale='dynamic')