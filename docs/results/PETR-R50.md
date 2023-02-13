<img src="../figs/logo.png" align="right" width="10%">

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

## PETR-R50

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Clean** | 0.3665 | 0.3174 | 0.8397 | 0.2796 | 0.6158 | 0.9543 | 0.2326 |
| **Motion Blur** | 0.2299    | 0.1378    | 0.9587     | 0.3164     | 0.8461     | 1.1190     | 0.2847     |
| **Color Quant** | 0.2472    | 0.1734    | 0.9121     | 0.3616     | 0.7807     | 1.1634     | 0.3473     |
| **Frame Lost** | 0.2166    | 0.0868    | 0.9513     | 0.3041     | 0.7597     | 1.0081     | 0.2629     |
| **Camera Crash** | 0.2320    | 0.1065    | 0.9383     | 0.2975     | 0.7220     | 1.0169     | 0.2585     |
| **Brightness** | 0.2841    | 0.2101    | 0.9049     | 0.3080     | 0.7429     | 1.0838     | 0.2552     |
| **Low Light** | 0.1877    | 0.0934    | 0.9190     | 0.3908     | 0.8423     | 1.4292     | 0.4372     |
| **Fog** | 0.2876    | 0.2161    | 0.9078     | 0.2928     | 0.7492     | 1.1781     | 0.2549     |
| **Snow** | 0.1417    | 0.0582    | 1.0437     | 0.4411     | 1.0177     | 1.3481     | 0.4713     |

## Experiment Log

Time: Fri Jan 20 23:29:31 2023

### Evaluating MotionBlur

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3239    | 0.2570    | 0.8781     | 0.2858     | 0.6727     | 0.9739     | 0.2356     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2018    | 0.0986    | 0.9766     | 0.3214     | 0.8868     | 1.1408     | 0.2901     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1641    | 0.0579    | 1.0215     | 0.3420     | 0.9787     | 1.2423     | 0.3283     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2299    | 0.1378    | 0.9587     | 0.3164     | 0.8461     | 1.1190     | 0.2847     |

### Evaluating Fog

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3029    | 0.2375    | 0.8984     | 0.2875     | 0.7251     | 1.1419     | 0.2470     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2860    | 0.2147    | 0.9097     | 0.2918     | 0.7539     | 1.1806     | 0.2582     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2739    | 0.1962    | 0.9153     | 0.2991     | 0.7687     | 1.2118     | 0.2595     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2876    | 0.2161    | 0.9078     | 0.2928     | 0.7492     | 1.1781     | 0.2549     |

### Evaluating Brightness

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3241    | 0.2669    | 0.8743     | 0.2872     | 0.7033     | 0.9941     | 0.2351     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2765    | 0.1997    | 0.9052     | 0.3117     | 0.7513     | 1.1250     | 0.2647     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2518    | 0.1637    | 0.9352     | 0.3251     | 0.7742     | 1.1324     | 0.2657     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2841    | 0.2101    | 0.9049     | 0.3080     | 0.7429     | 1.0838     | 0.2552     |

### Evaluating CameraCrash

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2798    | 0.1766    | 0.8893     | 0.2864     | 0.6690     | 1.0017     | 0.2403     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2100    | 0.0719    | 0.9598     | 0.2959     | 0.7433     | 1.0592     | 0.2600     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2061    | 0.0712    | 0.9658     | 0.3101     | 0.7538     | 0.9898     | 0.2752     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2320    | 0.1065    | 0.9383     | 0.2975     | 0.7220     | 1.0169     | 0.2585     |

### Evaluating FrameLost

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2886    | 0.1896    | 0.8823     | 0.2858     | 0.6668     | 0.9886     | 0.2386     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2020    | 0.0594    | 0.9510     | 0.3031     | 0.7563     | 1.0284     | 0.2666     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1594    | 0.0113    | 1.0206     | 0.3233     | 0.8559     | 1.0074     | 0.2836     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2166    | 0.0868    | 0.9513     | 0.3041     | 0.7597     | 1.0081     | 0.2629     |

Time: Sun Feb 12 11:13:23 2023

### Evaluating Snow

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2000    | 0.1137    | 0.9863     | 0.3239     | 0.8998     | 1.2796     | 0.3582     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1190    | 0.0317    | 1.0832     | 0.4815     | 1.1323     | 1.3518     | 0.4872     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1060    | 0.0292    | 1.0616     | 0.5178     | 1.0211     | 1.4129     | 0.5685     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1417    | 0.0582    | 1.0437     | 0.4411     | 1.0177     | 1.3481     | 0.4713     |

### Evaluating ColorQuant

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3411    | 0.2848    | 0.8517     | 0.2827     | 0.6436     | 0.9800     | 0.2553     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2664    | 0.1814    | 0.9047     | 0.2981     | 0.7528     | 1.0971     | 0.2874     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1341    | 0.0541    | 0.9799     | 0.5040     | 0.9458     | 1.4131     | 0.4993     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2472    | 0.1734    | 0.9121     | 0.3616     | 0.7807     | 1.1634     | 0.3473     |

### Evaluating LowLight

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2025    | 0.1046    | 0.9137     | 0.3686     | 0.8181     | 1.4262     | 0.3978     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2033    | 0.1051    | 0.9132     | 0.3690     | 0.8125     | 1.4162     | 0.3979     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1575    | 0.0704    | 0.9302     | 0.4349     | 0.8962     | 1.4453     | 0.5159     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1877    | 0.0934    | 0.9190     | 0.3908     | 0.8423     | 1.4292     | 0.4372     |
