#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVFormer/projects/configs/bevformer/bevformer_small_no_temp.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVFormer/bevformer_small_epoch_24.pth
GPUS=8
PORT=${PORT:-29501}

PYTHONPATH="/nvme/konglingdong/models/RoboDet/zoo/BEVFormer":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
    # -m debugpy --listen 5677 --wait-for-client 
