# Robust BEV Object Detection

## Getting Started

We benchmark prevalent camera-based object detection models under natural corruptions. Since models have different dependencies, please follow the official model repo to prepare the environment and data.

## Evaluate under corruption

To evaluate model under corruptions, add `corruptions` in mmdet config file, and run the folowwing command:
```shell
bash tools/dist_robust_test.sh
```
Results will be saved in `./log` folder with the prefix of model name.

## Acknowledgement
This project is built upon the following repo.
- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [DETR3D](https://github.com/WangYueFt/detr3d)
- [BEVDet](https://github.com/HuangJunJie2017/BEVDet)
- [BEVerse]()
- [SRCN3D]()
- [PolarFormer]()
- [ORA3D]()
