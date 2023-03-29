Modified from BEVDet [getting_started.md](https://github.com/HuangJunJie2017/BEVDet/blob/master/docs/getting_started.md)

# Prerequisites
Our code is tested on the following environment:
- Linux
- Python 3.8
- PyTorch 1.10.1
- Cudatoolkit 11.3.1 
- GCC 9.4.0
- MMCV==1.3.16
- MMDetection==2.14.0
- MMSegmentation==0.14.1
- MMDetection3d==0.17.2


# Installation
Please reference BEVDet's environment creation. A from-scratch script is as follows:
```shell
conda create -n solofusion python=3.8 -y
conda activate solofusion
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge -y
pip install mmcv-full==1.3.16 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
pip install -r requirements.txt
pip install -v -e .
pip install pillow==8.4.0 # Important! https://github.com/mit-han-lab/bevfusion/issues/63
pip install setuptools==59.5.0 # If you run into AttributeError: module 'distutils' has no attribute 'version'
```

# Data Preparation

**a. Please refer to [nuScenes](datasets/nuscenes_det.md) for initial preparation.**

```shell
python tools/create_data.py nuscenes --root-path ./data/nuscenes --out-dir ./data/nuscenes --extra-tag nuscenes
```
Note that L75-L82 in `tools/create_data.py` can be skipped; just generating nuscenes_infos_train.pkl & nuscenes_infos_val.pkl is enough
