# Log Folder

This folder contains the results of nuScenes-c benchmark. Some corruption types in [older version folder](./old_version/) are depreciated (e.g., ColorQuant, LowLight, Snow). Clean results are shown on the following table:

| **Model** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |  **Clean Config** |  **Crpt Config** | **Checkpoint** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |------- | ------- | ------- | 
|  **BEVFormer-Small**   |  0.5174    | 0.4164    | 0.6726     | 0.2734     | 0.3704     | 0.3941     | 0.1974     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_small.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/bevformer_small.py) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth) |
|  **BEVFormer-Small w/o temp**   | 0.4129    | 0.3461    | 0.7549     | 0.2832     | 0.4520     | 0.8917     | 0.2194     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_small_no_temp.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_small_epoch_24.pth) |
|  **BEVFormer-Base**   |  0.5174    | 0.4164    | 0.6726     | 0.2734     | 0.3704     | 0.3941     | 0.1974     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_base.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/bevformer_base.py) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth) |
|  **BEVFormer-Base w/o temp**   | 0.4129    | 0.3461    | 0.7549     | 0.2832     | 0.4520     | 0.8917     | 0.2194     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_base.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/bevformer_base_no_temp.py.py) | [ckpt](https://github.com/zhiqi-li/storage/releases/download/v1.0/bevformer_r101_dcn_24ep.pth) |
|  DETR   | | | | | | | | [config](../zoo/DETR3D//projects/configs/detr3d/detr3d_res101_gridmask.py) | [ckpt](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) |
|  DETR w cbgs  | | | | | | | | [config](../zoo/DETR3D//projects/configs/detr3d/detr3d_res101_gridmask_cbgs.py) | [config](../zoo/DETR3D//projects//configs/robust_test/detr3d_res101_gridmask_cbgs.py) | [ckpt](https://drive.google.com/file/d/1YWX-jIS6fxG5_JKUBNVcZtsPtShdjE4O/view?usp=sharing) |
|  PETR   | | | | | | | | | [config](../zoo/PETR/projects/configs/robust_test/) | TBD |
|  BEVDet    | | | | | | | |  | TBD | TBD |
|  BEVDepth   | | | | | | | |   | TBD | TBD |
|  BEVerse   | | | | | | | |   | [config](../zoo/BEVerse//projects//configs//robust_test/) | TBD |
|  ORA3D   | | | | | | | |  |  [config](../zoo/ora3d/projects/configs/robust_test/) | TBD |

The corruption results of PETRv2 might be unavailable, since it utilize data in `nuScenes/sweeps/` for temporal modeling, nuScenes-c only contains data in original `nuScenes/samples/` folder.