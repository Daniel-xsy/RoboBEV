# PolarFormer
<!-- TODO update the url to v2 -->
### [Paper](https://arxiv.org/abs/2206.15398)
> [**PolarFormer: Multi-camera 3D Object Detection
with Polar Transformers**](https://arxiv.org/abs/2206.15398),            
> Yanqin Jiang, Li Zhang, Zhenwei Miao, Xiatian Zhu, Jin Gao, Weiming Hu, Yu-Gang Jiang      

**This repository is an official implementation of PolarFormer.**
<div align="center">
  <img src="figs/pipeline.png"/>
</div><br/>

## Abstract

3D object detection in autonomous driving aims to reason “what” and “where” the objects of interest present in a 3D world. Following the conventional wisdom of previous 2D object detection, existing 3D object detection methods often adopt the canonical Cartesian coordinate system with perpendicular axis. However, we conjugate that this does not fit the nature of the ego car’s perspective, as each onboard camera perceives the world in shape of wedge intrinsic to the imaging geometry with radical (non-perpendicular) axis. Hence, in this paper we advocate the exploitation of the Polar coordinate system and propose a new Polar Transformer (PolarFormer) for more accurate 3D object detection in the bird’s-eye-view (BEV) taking as input only multi-camera 2D images. Specifically, we design a cross-attention based Polar detection head without restriction to the shape of input structure to deal with irregular Polar grids. For tackling the unconstrained object scale variations along Polar’s distance dimension, we further introduce a multiscale Polar representation learning strategy As a result, our model can make best use of the Polar representation rasterized via attending to the corresponding image observation in a sequence-to-sequence fashion subject to the geometric constraints. Thorough experiments on the nuScenes dataset demonstrate that our PolarFormer outperforms significantly state-of-the-art 3D object detection alternatives, as well as yielding competitive performance on BEV semantic segmentation task.

## News
- **(2022.11.25)**: Detection code of PolarFormer is released. <br>
- **(2022.7.1)**: The paper of PolarFomer is released on [arxiv](https://arxiv.org/abs/2206.15398).<br>
- **(2022.5.18)**: PolarFormer achieves state-of-the-art performance among the published works (**57.2% NDS** and **49.3% mAP**) on nuScenes 3D object detection [leaderboard](https://www.nuscenes.org/object-detection?externalData=all&mapData=no&modalities=Camera).<br>
- **(2022.5.16)**: PolarFormer-pure achieves state-of-the-art performance among the published works (**54.3% NDS** and **45.7% mAP**) on nuScenes 3D object detection (without external data) [leaderboard](https://www.nuscenes.org/object-detection?externalData=all&mapData=no&modalities=Camera).

## Get Started

### Environment
This implementation is build upon  [detr3d](https://github.com/WangYueFt/detr3d/blob/main/README.md), please follow the steps in [install.md](./docs/install.md) to prepare the environment.

### Data
Please follow the official instructions of mmdetection3d to process the nuScenes dataset.(https://mmdetection3d.readthedocs.io/en/v0.17.3/datasets/nuscenes_det.html)

After preparation, you will be able to see the following directory structure:  
  ```
  PolarFormer
  ├── mmdetection3d
  ├── projects
  │   ├── configs
  │   ├── mmdet3d_plugin
  ├── tools
  ├── data
  │   ├── nuscenes
  ├── ckpts
  ├── README.md
  ```
## Train & inference
```bash
cd PolarFormer
```
You can train the model following:
```bash
tools/dist_train.sh projects/configs/polarformer/polarformer_r101.py.py 8 --work-dir work_dirs/polarformer_r101/
```
You can evaluate the model following:
```bash
tools/dist_test.sh projects/configs/polarformer/polarformer_r101.py work_dirs/polarformer_r101/latest.pth 8 --eval bbox
```
## Main Results

### 3D Object Detection on nuScenes test set:
| model | mAP      | NDS     |
|:--------:|:----------:|:---------:|
PolarFormer, R101_DCN |41.5 |47.0 |
PolarFormer-T, R101_DCN|45.7 |54.3 |
PolarFormer, V2-99 |45.5 |50.3|
PolarFormer-T, V2-99 | 49.3|57.2|
<br>

### 3D Object Detection on nuScenes validation set:
| model | mAP      | NDS     | config | download |
|:--------:|:----------:|:---------:|:---------:|:---------:|
PolarFormer, R101_DCN| 39.6| 45.8| [config](projects/configs/polarformer/polarformer_r101.py) | [ckpt](https://drive.google.com/file/d/1Jgh49QJXls6XP6OAGhm744JHCGb7dGpP/view?usp=share_link) |
PolarFormer-w/o_bev_aug, R101_DCN |39.2 |46.0 | [config](projects/configs/polarformer/polarformer_r101_without_bev_aug.py) | [ckpt](https://drive.google.com/file/d/1GhCqJaaBEOYl-hkAwew2bmIt98AHPnpg/view?usp=share_link) / [log](https://drive.google.com/file/d/13hwLWauwTE9i2K2_-w8pNlfTKj9N1Jbl/view?usp=share_link)|
PolarFormer-T, R101_DCN| 43.2| 52.8| - | - |
PolarFormer, V2-99 |50.0 |56.2 |  [config](projects/configs/polarformer/polarformer_vovnet.py) | [ckpt](https://drive.google.com/file/d/1c5rgTpHA98dFKmQ9BJN0zZbSuBFT8_Bt/view?usp=share_link)|
<br>

**Note**: We adopt BEV data augmentation(random flipping, scaling and rotation) as the default setting when developing PolarFormer on nuScenes dataset. However, as the ablation in 2nd row indicates, BEV augmentation contributes little to the overall performance of PolarFormer. So please feel free to set "use_bev_aug = False" during training if you want to reduce computational burden.
### BEV Segmentation on nuScenes validation set:
| model | Drivable   | Crossing     | Walking    | Carpark     | Divider   |
|:--------:|:----------:|:---------:|:---------:|:---------:|:---------:|
PolarFormer, efficientnet-b0 | 81.0 | 48.9 | 55.8 | 52.6 | 42.2 |
PolarFormer-T, efficientnet-b0 | 82.6 | 54.3 | 59.4 | 56.7 | 46.2 |
PolarFormer-joint_det_seg, R101_DCN| 82.6 | 50.1 | 57.4 | 54.1 | 44.5 |
<br>

### Visualization
<div align="center">
  <img src="figs/visualization.png"/>
</div><br/>

## Reference
```bibtex   
@inproceedings{jiang2022polar,
  title={PolarFormer: Multi-camera 3D Object Detection with Polar Transformers},
  author={Jiang, Yanqin and Zhang, Li and Miao, Zhenwei and Zhu, Xiatian and Gao, Jin and Hu, Weiming and Jiang, Yu-Gang},
  booktitle={AAAI},
  year={2023}
}
```

## Acknowledgement
Many thanks to the following open-source projects:
* [mmdetection3d](https://github.com/open-mmlab/mmdetection3d)
* [detr3d](https://github.com/WangYueFt/detr3d)
