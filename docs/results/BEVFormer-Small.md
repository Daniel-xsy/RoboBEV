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

## BEVFormer-Small

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- |------- | ------- | ------- |------- | ------- | ------- |
| **Clean** |  0.4787    | 0.3700    | 0.7212     | 0.2792     | 0.4065     | 0.4364     | 0.2201     |
| **Motion Blur** | 0.2570    | 0.1344    | 0.8995     | 0.3264     | 0.6774     | 0.9625     | 0.2605     |
| **Color Quant** | 0.3275    | 0.2109    | 0.8476     | 0.2943     | 0.5234     | 0.8539     | 0.2601     |
| **Frame Lost** | 0.2459    | 0.0933    | 0.8959     | 0.3411     | 0.5742     | 0.9154     | 0.2804     |
| **Camera Crash** | 0.2771    | 0.1130    | 0.8627     | 0.3099     | 0.5398     | 0.8376     | 0.2446     |
| **Brightness** | 0.3741    | 0.2697    | 0.8064     | 0.2830     | 0.4796     | 0.8162     | 0.2226     |
| **Low Light** | 0.2851    | 0.1604    | 0.8643     | 0.3071     | 0.6088     | 0.9130     | 0.2573     |
| **Fog** | 0.3583    | 0.2486    | 0.8131     | 0.2862     | 0.5056     | 0.8301     | 0.2251     |
| **Snow** | 0.1809    | 0.0635    | 0.9630     | 0.3855     | 0.7741     | 1.1002     | 0.3863     |

## Experiment Log

Time: Mon Feb 13 13:47:14 2023

### Evaluating MotionBlur

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3582    | 0.2465    | 0.8195     | 0.2883     | 0.4981     | 0.8146     | 0.2304     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2246    | 0.0970    | 0.9206     | 0.3333     | 0.7192     | 1.0316     | 0.2657     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1883    | 0.0597    | 0.9583     | 0.3575     | 0.8148     | 1.0413     | 0.2853     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2570    | 0.1344    | 0.8995     | 0.3264     | 0.6774     | 0.9625     | 0.2605     |

### Evaluating Fog

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3711    | 0.2650    | 0.8033     | 0.2837     | 0.4920     | 0.8171     | 0.2176     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3604    | 0.2511    | 0.8082     | 0.2857     | 0.5050     | 0.8275     | 0.2246     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3433    | 0.2298    | 0.8279     | 0.2893     | 0.5197     | 0.8458     | 0.2332     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3583    | 0.2486    | 0.8131     | 0.2862     | 0.5056     | 0.8301     | 0.2251     |

### Evaluating Snow

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2212    | 0.0951    | 0.9511     | 0.3311     | 0.6783     | 1.0630     | 0.3032     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1648    | 0.0509    | 0.9654     | 0.4098     | 0.8067     | 1.0791     | 0.4246     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1567    | 0.0446    | 0.9724     | 0.4155     | 0.8374     | 1.1585     | 0.4310     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1809    | 0.0635    | 0.9630     | 0.3855     | 0.7741     | 1.1002     | 0.3863     |

### Evaluating ColorQuant

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3896    | 0.2884    | 0.7960     | 0.2806     | 0.4468     | 0.7878     | 0.2345     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3415    | 0.2247    | 0.8281     | 0.2868     | 0.5023     | 0.8339     | 0.2578     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2515    | 0.1197    | 0.9186     | 0.3156     | 0.6211     | 0.9401     | 0.2881     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3275    | 0.2109    | 0.8476     | 0.2943     | 0.5234     | 0.8539     | 0.2601     |

### Evaluating Brightness

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3936    | 0.2956    | 0.7911     | 0.2807     | 0.4517     | 0.7910     | 0.2273     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3735    | 0.2690    | 0.8093     | 0.2844     | 0.4798     | 0.8132     | 0.2237     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3551    | 0.2446    | 0.8188     | 0.2840     | 0.5073     | 0.8445     | 0.2168     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3741    | 0.2697    | 0.8064     | 0.2830     | 0.4796     | 0.8162     | 0.2226     |

### Evaluating LowLight

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3050    | 0.1800    | 0.8467     | 0.2986     | 0.5873     | 0.8714     | 0.2457     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3041    | 0.1809    | 0.8516     | 0.2981     | 0.5867     | 0.8758     | 0.2512     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2463    | 0.1202    | 0.8945     | 0.3245     | 0.6523     | 0.9919     | 0.2750     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2851    | 0.1604    | 0.8643     | 0.3071     | 0.6088     | 0.9130     | 0.2573     |

### Evaluating CameraCrash

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3295    | 0.1801    | 0.8284     | 0.2943     | 0.4946     | 0.7597     | 0.2285     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2664    | 0.0965    | 0.8986     | 0.3087     | 0.5365     | 0.8226     | 0.2524     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2353    | 0.0625    | 0.8611     | 0.3266     | 0.5884     | 0.9304     | 0.2530     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2771    | 0.1130    | 0.8627     | 0.3099     | 0.5398     | 0.8376     | 0.2446     |

### Evaluating FrameLost

#### Severity easy

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.3283    | 0.1947    | 0.8296     | 0.2923     | 0.4934     | 0.8405     | 0.2350     |

#### Severity mid

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2378    | 0.0684    | 0.9013     | 0.3229     | 0.5732     | 0.9090     | 0.2576     |

#### Severity hard

Evaluating Results

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.1717    | 0.0167    | 0.9569     | 0.4081     | 0.6559     | 0.9968     | 0.3486     |

#### Average

| **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| ------- | ------- | -------- | -------- | -------- | -------- | -------- |
| 0.2459    | 0.0933    | 0.8959     | 0.3411     | 0.5742     | 0.9154     | 0.2804     |
