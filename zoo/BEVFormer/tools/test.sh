CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m pdb ./tools/test.py \
projects/configs/bevformer/bevformer_base.py \
ckpts/bevformer_r101_dcn_24ep.pth \
--eval bbox \