# Copyright (c) Phigent Robotics. All rights reserved.

_base_ = ['./bevdet-r50.py']

fp16 = dict(loss_scale='dynamic')