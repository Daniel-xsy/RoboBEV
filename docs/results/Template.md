<img src="../figs/logo.png" align="right" width="34%">

# RoboDet Benchmark

The official nuScenes metric are considered in our benchmark:

- **Average Translation Error (ATE)** is the Euclidean center distance in 2D (units in meters). 
- **Average Scale Error (ASE)** is the 3D intersection-over-union (IoU) after aligning orientation and translation (1 − IoU).
- **Average Orientation Error (AOE)** is the smallest yaw angle difference between prediction and ground truth (radians). All angles are measured on a full 360-degree period except for barriers where they are measured on a 180-degree period.
- **Average Velocity Error (AVE)** is the absolute velocity error as the L2 norm of the velocity differences in 2D (m/s).
- **Average Attribute Error (AAE)** is defined as 1 minus attribute classification accuracy (1 − acc).
- **nuScenes Detection Score (NDS)**: $$\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))]$$

## ModleName

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Clean** |  |  |  |  |  |  |  |
| **Motion Blur** |  |  |  |  |  |  |  |
| **Color Quant** |  |  |  |  |  |  |  |
| **Frame Lost** |  |  |  |  |  |  |  |
| **Camera Crash** |  |  |  |  |  |  |  |
| **Brightness** |  |  |  |  |  |  |  |
| **Low Light** |  |  |  |  |  |  |  |
| **Fog** |  |  |  |  |  |  |  |
| **Snow** |  |  |  |  |  |  |  |

## Motion Blur

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |

## Color Quant

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |

## Frame Lost

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |

## Camera Crash

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |

## Brightness

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |

## Low Light

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |

## Fog

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |

## Snow

| **Level** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Easy** |  |  |  |  |  |  |  |
| **Mid** |  |  |  |  |  |  |  |
| **Hard** |  |  |  |  |  |  |  |
