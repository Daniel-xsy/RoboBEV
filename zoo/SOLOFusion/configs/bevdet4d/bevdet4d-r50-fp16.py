# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['./bevdet4d-r50.py']

fp16 = dict(loss_scale='dynamic')