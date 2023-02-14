<img src="../figs/logo2.png" align="right" width="30%">

# RoboDet Benchmark

The official [nuScenes metric](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any) are considered in our benchmark:

### Average Precision (AP)

The average precision (AP) defines a match by thresholding the 2D center distance d on the ground plane instead of the intersection over union (IoU). This is done in order to decouple detection from object size and orientation but also because objects with small footprints, like pedestrians and bikes, if detected with a small translation error, give 0 IoU.
We then calculate AP as the normalized area under the precision-recall curve for recall and precision over 10%. Operating points where recall or precision is less than 10% are removed in order to minimize the impact of noise commonly seen in low precision and recall regions. If no operating point in this region is achieved, the AP for that class is set to zero. We then average over-matching thresholds of $\mathbb{D}=\{0.5, 1, 2, 4\}$ meters and the set of classes $\mathbb{C}$ :

$$
\text{mAP}= \frac{1}{|\mathbb{C}||\mathbb{D}|}\sum_{c\in\mathbb{C}}\sum_{d\in\mathbb{D}}\text{AP}_{c,d}
$$

### True Positive (TP)

All TP metrics are calculated using $d=2$ m center distance during matching, and they are all designed to be positive scalars. Matching and scoring happen independently per class and each metric is the average of the cumulative mean at each achieved recall level above 10%. If a 10% recall is not achieved for a particular class, all TP errors for that class are set to 1. 

- **Average Translation Error (ATE)** is the Euclidean center distance in 2D (units in meters). 
- **Average Scale Error (ASE)** is the 3D intersection-over-union (IoU) after aligning orientation and translation (1 − IoU).
- **Average Orientation Error (AOE)** is the smallest yaw angle difference between prediction and ground truth (radians). All angles are measured on a full 360-degree period except for barriers where they are measured on a 180-degree period.
- **Average Velocity Error (AVE)** is the absolute velocity error as the L2 norm of the velocity differences in 2D (m/s).
- **Average Attribute Error (AAE)** is defined as 1 minus attribute classification accuracy (1 − acc).

### nuScenes Detection Score (NDS)
mAP with a threshold on IoU is perhaps the most popular metric for object detection. However, this metric can not capture all aspects of the nuScenes detection tasks, like velocity and attribute estimation. Further, it couples location, size, and orientation estimates. nuScenes proposed instead consolidating the different error types into a scalar score:

$$
\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))]
$$

## BEVFormer-Small-SingleFrame

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Clean** | 0.2622    | 0.1324    | 0.9352     | 0.3024     | 0.5556     | 1.1106     | 0.2466     |
| **Motion Blur** | 0.2570    | 0.1344    | 0.8995     | 0.3264     | 0.6774     | 0.9625     | 0.2605     |
| **Color Quant** | 0.3275    | 0.2109    | 0.8476     | 0.2943     | 0.5234     | 0.8539     | 0.2601     |
| **Frame Lost** | 0.2459    | 0.0933    | 0.8959     | 0.3411     | 0.5742     | 0.9154     | 0.2804     |
| **Camera Crash** | 0.2771    | 0.1130    | 0.8627     | 0.3099     | 0.5398     | 0.8376     | 0.2446     |
| **Brightness** | 0.3741    | 0.2697    | 0.8064     | 0.2830     | 0.4796     | 0.8162     | 0.2226     |
| **Low Light** | 0.2851    | 0.1604    | 0.8643     | 0.3071     | 0.6088     | 0.9130     | 0.2573     |
| **Fog** | 0.3583    | 0.2486    | 0.8131     | 0.2862     | 0.5056     | 0.8301     | 0.2251     |
| **Snow** | 0.1809    | 0.0635    | 0.9630     | 0.3855     | 0.7741     | 1.1002     | 0.3863     |

## Experiment Log
