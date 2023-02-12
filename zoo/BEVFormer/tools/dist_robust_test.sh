#!/usr/bin/env bash

CONFIG=/nvme/konglingdong/models/RoboDet/zoo/BEVFormer/projects/configs/robust_test/bevformer_base_no_temp.py
CHECKPOINT=/nvme/konglingdong/models/RoboDet/models/BEVFormer/bevformer_r101_dcn_24ep.pth
GPUS=8
PORT=${PORT:-29501}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/robust_test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4} --eval bbox
