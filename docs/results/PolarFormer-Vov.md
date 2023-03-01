<img src="../figs/logo2.png" align="right" width="30%">

# RoboBEV Benchmark

The official [nuScenes metrics](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any) are considered in our benchmark:

### Average Precision (AP)

The average precision (AP) defines a match by thresholding the 2D center distance d on the ground plane instead of the intersection over union (IoU). This is done in order to decouple detection from object size and orientation but also because objects with small footprints, like pedestrians and bikes, if detected with a small translation error, give $0$ IoU.
We then calculate AP as the normalized area under the precision-recall curve for recall and precision over 10%. Operating points where recall or precision is less than $10$% are removed in order to minimize the impact of noise commonly seen in low precision and recall regions. If no operating point in this region is achieved, the AP for that class is set to zero. We then average over-matching thresholds of $\mathbb{D}=\{0.5, 1, 2, 4\}$ meters and the set of classes $\mathbb{C}$ :

$$
\text{mAP}= \frac{1}{|\mathbb{C}||\mathbb{D}|}\sum_{c\in\mathbb{C}}\sum_{d\in\mathbb{D}}\text{AP}_{c,d} .
$$

### True Positive (TP)

All TP metrics are calculated using $d=2$ m center distance during matching, and they are all designed to be positive scalars. Matching and scoring happen independently per class and each metric is the average of the cumulative mean at each achieved recall level above $10$%. If a $10$% recall is not achieved for a particular class, all TP errors for that class are set to $1$. 

- **Average Translation Error (ATE)** is the Euclidean center distance in 2D (units in meters). 
- **Average Scale Error (ASE)** is the 3D intersection-over-union (IoU) after aligning orientation and translation ($1$ − IoU).
- **Average Orientation Error (AOE)** is the smallest yaw angle difference between prediction and ground truth (radians). All angles are measured on a full $360$-degree period except for barriers where they are measured on a $180$-degree period.
- **Average Velocity Error (AVE)** is the absolute velocity error as the L2 norm of the velocity differences in 2D (m/s).
- **Average Attribute Error (AAE)** is defined as $1$ minus attribute classification accuracy ($1$ − acc).

### nuScenes Detection Score (NDS)

mAP with a threshold on IoU is perhaps the most popular metric for object detection. However, this metric can not capture all aspects of the nuScenes detection tasks, like velocity and attribute estimation. Further, it couples location, size, and orientation estimates. nuScenes proposed instead consolidating the different error types into a scalar score:

$$
\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))] .
$$


## PolarFormer-Vov

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4558  | 0.4028  | 0.7097  | 0.2690 | 0.4019  | 0.8682  | 0.2072  |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.3135    | 0.1453    | 0.7626     | 0.2815     | 0.4519     | 0.8735     | 0.2216     |
|   Frame Lost   | 0.2811    | 0.1155    | 0.8019     | 0.3015     | 0.4956     | 0.9158     | 0.2512     |
|  Color Quant   | 0.3076    | 0.2000    | 0.8846     | 0.2962     | 0.5393     | 1.0044     | 0.2483     |
|  Motion Blur   | 0.2344    | 0.1256    | 0.9392     | 0.3616     | 0.6840     | 1.0992     | 0.3489     |
|   Brightness   | 0.4280    | 0.3619    | 0.7447     | 0.2696     | 0.4413     | 0.8667     | 0.2065     |
|   Low Light    | 0.2441    | 0.1361    | 0.8828     | 0.3647     | 0.6506     | 1.2090     | 0.3419     |
|      Fog       | 0.4061    | 0.3349    | 0.7651     | 0.2743     | 0.4487     | 0.9100     | 0.2156     |
|      Snow      | 0.2468    | 0.1384    | 0.9104     | 0.3375     | 0.6427     | 1.1737     | 0.3337     |


## Experiment Log

> Time: Tue Feb 28 20:21:10 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3642    | 0.2184    | 0.7372     | 0.2751     | 0.3895     | 0.8415     | 0.2062     |
|   Moderate   | 0.2874    | 0.1044    | 0.7871     | 0.2791     | 0.4339     | 0.9203     | 0.2273     |
|     Hard     | 0.2889    | 0.1131    | 0.7634     | 0.2903     | 0.5323     | 0.8587     | 0.2313     |
|   Average    | 0.3135    | 0.1453    | 0.7626     | 0.2815     | 0.4519     | 0.8735     | 0.2216     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3737    | 0.2476    | 0.7386     | 0.2731     | 0.4146     | 0.8657     | 0.2091     |
|   Moderate   | 0.2691    | 0.0823    | 0.8046     | 0.2836     | 0.5105     | 0.8981     | 0.2241     |
|     Hard     | 0.2007    | 0.0165    | 0.8625     | 0.3477     | 0.5618     | 0.9837     | 0.3203     |
|   Average    | 0.2811    | 0.1155    | 0.8019     | 0.3015     | 0.4956     | 0.9158     | 0.2512     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4076    | 0.3313    | 0.7586     | 0.2741     | 0.4406     | 0.8913     | 0.2161     |
|   Moderate   | 0.3115    | 0.2053    | 0.8721     | 0.2874     | 0.5061     | 0.9983     | 0.2480     |
|     Hard     | 0.2037    | 0.0632    | 1.0230     | 0.3271     | 0.6711     | 1.1235     | 0.2809     |
|   Average    | 0.3076    | 0.2000    | 0.8846     | 0.2962     | 0.5393     | 1.0044     | 0.2483     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3917    | 0.3087    | 0.7761     | 0.2725     | 0.4560     | 0.8936     | 0.2281     |
|   Moderate   | 0.1837    | 0.0485    | 1.0086     | 0.3361     | 0.7545     | 1.0995     | 0.3150     |
|     Hard     | 0.1277    | 0.0197    | 1.0328     | 0.4761     | 0.8415     | 1.3045     | 0.5037     |
|   Average    | 0.2344    | 0.1256    | 0.9392     | 0.3616     | 0.6840     | 1.0992     | 0.3489     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4531    | 0.3953    | 0.7128     | 0.2678     | 0.4059     | 0.8563     | 0.2025     |
|   Moderate   | 0.4306    | 0.3635    | 0.7436     | 0.2697     | 0.4357     | 0.8587     | 0.2039     |
|     Hard     | 0.4004    | 0.3268    | 0.7778     | 0.2713     | 0.4823     | 0.8852     | 0.2130     |
|   Average    | 0.4280    | 0.3619    | 0.7447     | 0.2696     | 0.4413     | 0.8667     | 0.2065     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3209    | 0.2171    | 0.8206     | 0.2889     | 0.5348     | 1.0134     | 0.2323     |
|   Moderate   | 0.2590    | 0.1336    | 0.8943     | 0.3233     | 0.5944     | 1.1492     | 0.2660     |
|     Hard     | 0.1523    | 0.0577    | 0.9334     | 0.4818     | 0.8226     | 1.4644     | 0.5275     |
|   Average    | 0.2441    | 0.1361    | 0.8828     | 0.3647     | 0.6506     | 1.2090     | 0.3419     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4240    | 0.3580    | 0.7499     | 0.2724     | 0.4316     | 0.8857     | 0.2109     |
|   Moderate   | 0.4049    | 0.3356    | 0.7635     | 0.2741     | 0.4520     | 0.9230     | 0.2166     |
|     Hard     | 0.3893    | 0.3110    | 0.7820     | 0.2763     | 0.4626     | 0.9212     | 0.2194     |
|   Average    | 0.4061    | 0.3349    | 0.7651     | 0.2743     | 0.4487     | 0.9100     | 0.2156     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3341    | 0.2416    | 0.8216     | 0.2870     | 0.5129     | 0.9977     | 0.2476     |
|   Moderate   | 0.2294    | 0.1045    | 0.9493     | 0.3203     | 0.6473     | 1.1993     | 0.3114     |
|     Hard     | 0.1769    | 0.0690    | 0.9604     | 0.4053     | 0.7680     | 1.3241     | 0.4421     |
|   Average    | 0.2468    | 0.1384    | 0.9104     | 0.3375     | 0.6427     | 1.1737     | 0.3337     |



## References

```bib
@article{jiang2022polarformer,
  title={Polarformer: Multi-camera 3d object detection with polar transformers},
  author={Jiang, Yanqin and Zhang, Li and Miao, Zhenwei and Zhu, Xiatian and Gao, Jin and Hu, Weiming and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2206.15398},
  year={2022}
}
```