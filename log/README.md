# Log Folder

This folder contains the results of nuScenes-c benchmark. Some corruption types in [older version folder](./old_version/) are depreciated (e.g., ColorQuant, LowLight, Snow). Clean results will be add later.

| **Model** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |  **Clean Config** |  **Corruption Config** | 
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |------- | ------- | 
|  BEVFormer-Base   |  0.5174    | 0.4164    | 0.6726     | 0.2734     | 0.3704     | 0.3941     | 0.1974     | [config](../zoo/BEVFormer/projects/configs/bevformer/bevformer_base.py) | [config](../zoo/BEVFormer/projects/configs/robust_test/bevformer_base.py) |
|  DETR   | | | | | | | | | [config](../zoo/DETR3D//projects//configs//robust_test/) |
|  PETR   | | | | | | | | | [config](../zoo/PETR/projects/configs/robust_test/) |
|  BEVDet    | | | | | | | |  | TBD |
|  BEVDepth   | | | | | | | |   | TBD |
|  BEVerse   | | | | | | | |   | [config](../zoo/BEVerse//projects//configs//robust_test/) |
|  ORA3D   | | | | | | | |  |  [config](../zoo/ora3d/projects/configs/robust_test/) |

The corruption results of PETRv2 might be unavailable, since it utilize data in `nuScenes/sweeps/` for temporal modeling, nuScenes-c only contains data in original `nuScenes/samples/` folder.