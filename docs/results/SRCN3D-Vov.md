<img src="..\figs\logo2.png" align="right" width="30%">

# RoboBEV Benchmark

The official [nuScenes metrics](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any) are considered in our benchmark:

### Average Precision (AP)

The average precision (AP) defines a match by thresholding the 2D center distance d on the ground plane instead of the intersection over union (IoU). This is done in order to decouple detection from object size and orientation but also because objects with small footprints, like pedestrians and bikes, if detected with a small translation error, give $0$ IoU.
We then calculate AP as the normalized area under the precision-recall curve for recall and precision over 10%. Operating points where recall or precision is less than $10$% are removed in order to minimize the impact of noise commonly seen in low precision and recall regions. If no operating point in this region is achieved, the AP for that class is set to zero. We then average over-matching thresholds of $\mathbb{D}=\{0.5, 1, 2, 4\}$ meters and the set of classes $\mathbb{C}$ :

$$
\text{mAP}= \frac{1}{|\mathbb{C}||\mathbb{D}|}\sum_{c\in\mathbb{C}}\sum_{d\in\mathbb{D}}\text{AP}_{c,d} .
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
\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))] .
$$


## SRCN3D-Vov

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4205 | 0.3475 | 0.7855 | 0.2994 | 0.4099 | 0.8352 | 0.2030 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.2875    | 0.1252    | 0.8435     | 0.3139     | 0.4879     | 0.8897     | 0.2165     |
|   Frame Lost   | 0.2579    | 0.0982    | 0.8710     | 0.3428     | 0.5324     | 0.9194     | 0.2458     |
|  Color Quant   | 0.2827    | 0.1755    | 0.9167     | 0.3443     | 0.5574     | 1.0077     | 0.2747     |
|  Motion Blur   | 0.2143    | 0.1102    | 0.9833     | 0.3966     | 0.7434     | 1.1151     | 0.3500     |
|   Brightness   | 0.3886    | 0.3086    | 0.8175     | 0.3018     | 0.4660     | 0.8720     | 0.2001     |
|   Low Light    | 0.2274    | 0.1142    | 0.9192     | 0.3866     | 0.6475     | 1.2095     | 0.3435     |
|      Fog       | 0.3774    | 0.2911    | 0.8227     | 0.3045     | 0.4646     | 0.8864     | 0.2034     |
|      Snow      | 0.2499    | 0.1418    | 0.9299     | 0.3575     | 0.6125     | 1.1351     | 0.3176     |


## Experiment Log

> Time: Tue Feb 28 20:20:43 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3344    | 0.1920    | 0.8221     | 0.3077     | 0.4231     | 0.8545     | 0.2083     |
|   Moderate   | 0.2572    | 0.0882    | 0.8759     | 0.3163     | 0.4853     | 0.9639     | 0.2283     |
|     Hard     | 0.2709    | 0.0955    | 0.8324     | 0.3177     | 0.5552     | 0.8506     | 0.2128     |
|   Average    | 0.2875    | 0.1252    | 0.8435     | 0.3139     | 0.4879     | 0.8897     | 0.2165     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3451    | 0.2131    | 0.8110     | 0.3056     | 0.4247     | 0.8696     | 0.2033     |
|   Moderate   | 0.2490    | 0.0687    | 0.8758     | 0.3279     | 0.5323     | 0.9026     | 0.2155     |
|     Hard     | 0.1797    | 0.0126    | 0.9262     | 0.3949     | 0.6403     | 0.9861     | 0.3186     |
|   Average    | 0.2579    | 0.0982    | 0.8710     | 0.3428     | 0.5324     | 0.9194     | 0.2458     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3727    | 0.2864    | 0.8238     | 0.3029     | 0.4619     | 0.9057     | 0.2102     |
|   Moderate   | 0.2903    | 0.1741    | 0.9004     | 0.3177     | 0.5228     | 0.9912     | 0.2351     |
|     Hard     | 0.1851    | 0.0660    | 1.0260     | 0.4123     | 0.6876     | 1.1261     | 0.3789     |
|   Average    | 0.2827    | 0.1755    | 0.9167     | 0.3443     | 0.5574     | 1.0077     | 0.2747     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3563    | 0.2671    | 0.8508     | 0.3051     | 0.4935     | 0.9035     | 0.2192     |
|   Moderate   | 0.1731    | 0.0443    | 1.0448     | 0.3690     | 0.8149     | 1.1093     | 0.3071     |
|     Hard     | 0.1135    | 0.0192    | 1.0544     | 0.5158     | 0.9219     | 1.3326     | 0.5237     |
|   Average    | 0.2143    | 0.1102    | 0.9833     | 0.3966     | 0.7434     | 1.1151     | 0.3500     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4164    | 0.3403    | 0.7888     | 0.2996     | 0.4180     | 0.8351     | 0.1961     |
|   Moderate   | 0.3900    | 0.3104    | 0.8091     | 0.3006     | 0.4656     | 0.8780     | 0.1987     |
|     Hard     | 0.3593    | 0.2752    | 0.8547     | 0.3053     | 0.5144     | 0.9030     | 0.2055     |
|   Average    | 0.3886    | 0.3086    | 0.8175     | 0.3018     | 0.4660     | 0.8720     | 0.2001     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2955    | 0.1823    | 0.8712     | 0.3198     | 0.5254     | 1.0441     | 0.2401     |
|   Moderate   | 0.2420    | 0.1145    | 0.9313     | 0.3446     | 0.6051     | 1.1403     | 0.2719     |
|     Hard     | 0.1448    | 0.0459    | 0.9552     | 0.4955     | 0.8121     | 1.4442     | 0.5184     |
|   Average    | 0.2274    | 0.1142    | 0.9192     | 0.3866     | 0.6475     | 1.2095     | 0.3435     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3926    | 0.3114    | 0.8108     | 0.3016     | 0.4482     | 0.8693     | 0.2014     |
|   Moderate   | 0.3767    | 0.2923    | 0.8234     | 0.3050     | 0.4749     | 0.8881     | 0.2028     |
|     Hard     | 0.3629    | 0.2697    | 0.8338     | 0.3068     | 0.4707     | 0.9017     | 0.2060     |
|   Average    | 0.3774    | 0.2911    | 0.8227     | 0.3045     | 0.4646     | 0.8864     | 0.2034     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3127    | 0.2104    | 0.8756     | 0.3163     | 0.5154     | 0.9781     | 0.2398     |
|   Moderate   | 0.2486    | 0.1306    | 0.9398     | 0.3351     | 0.6059     | 1.1629     | 0.2858     |
|     Hard     | 0.1883    | 0.0845    | 0.9742     | 0.4211     | 0.7163     | 1.2643     | 0.4273     |
|   Average    | 0.2499    | 0.1418    | 0.9299     | 0.3575     | 0.6125     | 1.1351     | 0.3176     |s



## References

```bib
@article{shi2022srcn3d,
  title={Srcn3d: Sparse r-cnn 3d surround-view camera object detection and tracking for autonomous driving},
  author={Shi, Yining and Shen, Jingyan and Sun, Yifan and Wang, Yunlong and Li, Jiaxin and Sun, Shiqi and Jiang, Kun and Yang, Diange},
  journal={arXiv preprint arXiv:2206.14451},
  year={2022}
}
```