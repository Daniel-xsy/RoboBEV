#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/FCOS3D/projects/config/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/FCOS3D/fcos3d_r101_caffe_fpn_gn-head_dcn_2x8_1x_nus-mono3d_finetune_20210717_095645-8d806dc2.pth
GPUS=8
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
# -m debugpy --listen 5680 --wait-for-client 