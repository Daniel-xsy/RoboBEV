CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m debugpy --listen 5678 --wait-for-client ./tools/test.py \
projects/configs/bevformer/bevformer_base.py \
../../models/BEVFormer/bevformer_r101_dcn_24ep.pth \
--eval bbox \
# --corruption_test \