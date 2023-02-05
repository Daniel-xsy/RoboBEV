# Log Folder

This folder contains the results of nuScenes-c benchmark. Some corruption types in [older version folder](./old_version/) are depreciated (e.g., `ColorQuant`, `LowLight`, `Snow`). New results are added under this folder.

Clean results are shown on the following table:

| **Model** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |  **Clean Config** |  **Crpt Config** | **Ckpt** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |------- | ------- | ------- | 
|  **BEVFormer-Small**   |  0.4787    | 0.3700    | 0.7212     | 0.2792     | 0.4065     | 0.4364     | 0.2201     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_small.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/bevformer_small.py) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth) |
|  **BEVFormer-Small w/o temp**   | 0.2622    | 0.1324    | 0.9352     | 0.3024     | 0.5556     | 1.1106     | 0.2466     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_small_no_temp.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth) |
|  **BEVFormer-Base**   |  0.5174    | 0.4164    | 0.6726     | 0.2734     | 0.3704     | 0.3941     | 0.1974     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_base.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/bevformer_base.py) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth) |
|  **BEVFormer-Base w/o temp**   | 0.4129    | 0.3461    | 0.7549     | 0.2832     | 0.4520     | 0.8917     | 0.2194     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_base.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/bevformer_base_no_temp.py.py) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth) |
|  **DETR**   | | | | | | | | [config](../zoo/DETR3D//projects/configs/detr3d/detr3d_res101_gridmask.py) | [config](../zoo/DETR3D//projects/configs/robust_test/detr3d_res101_gridmask.py) |[ckpt](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) |
|  **DETR w cbgs**  | | | | | | | | [config](../zoo/DETR3D//projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py) | [config](../zoo/DETR3D//projects//configs/robust_test/detr3d_res101_gridmask_cbgs.py) | [ckpt](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) |
|  **PETR-r50-p4**   | | | | | | | | [config](../zoo/PETR/projects/configs/petr/petr_r50dcn_gridmask_p4.py) | [config](../zoo/PETR/projects/configs/robust_test/petr_r50dcn_gridmask_p4.py) | (ckpt)[https://drive.google.com/file/d/1eYymeIbS0ecHhQcB8XAFazFxLPm3wIHY/view?usp=sharing] |
|  **PETR-vov-p4**   | | | | | | | | [config](../zoo/PETR/projects/configs/petr/petr_vovnet_gridmask_p4_1600x640.py) | [config](../zoo/PETR/projects/configs/robust_test/petr_vovnet_gridmask_p4_1600x640.py) | (ckpt)[https://drive.google.com/file/d/1SV0_n0PhIraEXHJ1jIdMu3iMg9YZsm8c/view?usp=sharing] |
|  BEVDet    | | | | | | | |  | TBD | TBD |
|  BEVDepth   | | | | | | | |   | TBD | TBD |
|  BEVerse   | | | | | | | |   | [config](../zoo/BEVerse//projects//configs//robust_test/) | TBD |
|  ORA3D   | | | | | | | |  |  [config](../zoo/ora3d/projects/configs/robust_test/) | TBD |

### Notice
The corruption results of PETRv2 might be unavailable, since it utilize data in `nuScenes/sweeps/` for temporal modeling, nuScenes-c only contains data in original `nuScenes/samples/` folder.