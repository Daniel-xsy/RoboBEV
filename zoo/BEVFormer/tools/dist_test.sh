#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVFormer/projects/configs/robust_test/bevformer_base.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVFormer/bevformer_r101_dcn_24ep.pth
GPUS=1
PORT=${PORT:-29503}

PYTHONPATH="/nvme/konglingdong/models/RoboDet/zoo/BEVFormer":$PYTHONPATH \
python -m debugpy --listen 5677 --wait-for-client -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
