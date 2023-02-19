<img src="F:\Research\Robust BEV Detection\Robust-BEV-Detection\docs\figs\logo2.png" align="right" width="30%">

# RoboDet Benchmark

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


## BEVerse-Tiny

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4665 | 0.3214  | 0.6807 | 0.2782 | 0.4657 | 0.3281 | 0.1893 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.3181    | 0.1218    | 0.7447     | 0.3545     | 0.5479     | 0.4974     | 0.2833     |
|   Frame Lost   | 0.3037    | 0.1466    | 0.7892     | 0.3511     | 0.6217     | 0.6491     | 0.2844     |
|  Color Quant   | 0.2600    | 0.1497    | 0.8577     | 0.4758     | 0.6711     | 0.6931     | 0.4676     |
|  Motion Blur   | 0.2647    | 0.1456    | 0.8139     | 0.4269     | 0.6275     | 0.8103     | 0.4225     |
|   Brightness   | 0.2656    | 0.1512    | 0.8120     | 0.4548     | 0.6799     | 0.7029     | 0.4507     |
|   Low Light    | 0.0593    | 0.0235    | 0.9744     | 0.7926     | 0.9961     | 0.9437     | 0.8304     |
|      Fog       | 0.2781    | 0.1348    | 0.8467     | 0.3967     | 0.6135     | 0.6596     | 0.3764     |
|      Snow      | 0.0644    | 0.0251    | 0.9662     | 0.7966     | 0.8893     | 0.9829     | 0.8464     |


## Experiment Log

> Time: Fri Jan 27 17:36:37 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3886    | 0.1928    | 0.7212     | 0.2857     | 0.5062     | 0.3777     | 0.1876     |
|   Moderate   | 0.2973    | 0.0890    | 0.7908     | 0.3478     | 0.5551     | 0.5042     | 0.2746     |
|     Hard     | 0.2685    | 0.0834    | 0.7221     | 0.4301     | 0.5823     | 0.6103     | 0.3878     |
|   Average    | 0.3181    | 0.1218    | 0.7447     | 0.3545     | 0.5479     | 0.4974     | 0.2833     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4131    | 0.2628    | 0.7314     | 0.2878     | 0.5277     | 0.4371     | 0.1990     |
|   Moderate   | 0.3025    | 0.1287    | 0.8078     | 0.3129     | 0.6440     | 0.6379     | 0.2159     |
|     Hard     | 0.1956    | 0.0482    | 0.8283     | 0.4527     | 0.6934     | 0.8722     | 0.4384     |
|   Average    | 0.3037    | 0.1466    | 0.7892     | 0.3511     | 0.6217     | 0.6491     | 0.2844     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4349    | 0.2802    | 0.7130     | 0.2810     | 0.4865     | 0.3622     | 0.2099     |
|   Moderate   | 0.2760    | 0.1487    | 0.8523     | 0.4235     | 0.6118     | 0.6739     | 0.4225     |
|     Hard     | 0.0692    | 0.0201    | 1.0079     | 0.7228     | 0.9151     | 1.0433     | 0.7703     |
|   Average    | 0.2600    | 0.1497    | 0.8577     | 0.4758     | 0.6711     | 0.6931     | 0.4676     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4139    | 0.2568    | 0.7438     | 0.2867     | 0.4955     | 0.4274     | 0.1917     |
|   Moderate   | 0.2281    | 0.1131    | 0.8005     | 0.4239     | 0.6969     | 0.9409     | 0.4229     |
|     Hard     | 0.1523    | 0.0668    | 0.8975     | 0.5702     | 0.6900     | 1.0627     | 0.6530     |
|   Average    | 0.2647    | 0.1456    | 0.8139     | 0.4269     | 0.6275     | 0.8103     | 0.4225     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3697    | 0.2385    | 0.7157     | 0.3664     | 0.5749     | 0.5122     | 0.3269     |
|   Moderate   | 0.2623    | 0.1368    | 0.8140     | 0.4242     | 0.6759     | 0.7325     | 0.4148     |
|     Hard     | 0.1648    | 0.0783    | 0.9064     | 0.5737     | 0.7889     | 0.8639     | 0.6103     |
|   Average    | 0.2656    | 0.1512    | 0.8120     | 0.4548     | 0.6799     | 0.7029     | 0.4507     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0719    | 0.0375    | 0.9701     | 0.7919     | 0.9588     | 0.9258     | 0.8223     |
|   Moderate   | 0.0591    | 0.0242    | 0.9748     | 0.7930     | 0.9916     | 0.9409     | 0.8296     |
|     Hard     | 0.0469    | 0.0087    | 0.9784     | 0.7929     | 1.0379     | 0.9644     | 0.8392     |
|   Average    | 0.0674    | 0.0330    | 0.9721     | 0.7923     | 0.9705     | 0.9314     | 0.8249     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3297    | 0.1721    | 0.8259     | 0.3396     | 0.5572     | 0.5544     | 0.2863     |
|   Moderate   | 0.2655    | 0.1323    | 0.8435     | 0.4215     | 0.6340     | 0.6928     | 0.4147     |
|     Hard     | 0.2392    | 0.1001    | 0.8707     | 0.4289     | 0.6493     | 0.7316     | 0.4281     |
|   Average    | 0.2781    | 0.1348    | 0.8467     | 0.3967     | 0.6135     | 0.6596     | 0.3764     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1157    | 0.0463    | 0.9584     | 0.6531     | 0.7724     | 0.9625     | 0.7278     |
|   Moderate   | 0.0381    | 0.0116    | 0.9676     | 0.8715     | 0.9468     | 0.9868     | 0.9046     |
|     Hard     | 0.0394    | 0.0173    | 0.9727     | 0.8653     | 0.9487     | 0.9993     | 0.9067     |
|   Average    | 0.0644    | 0.0251    | 0.9662     | 0.7966     | 0.8893     | 0.9829     | 0.8464     |



## References

```bib
@article{zhang2022beverse,
  title={Beverse: Unified perception and prediction in birds-eye-view for vision-centric autonomous driving},
  author={Zhang, Yunpeng and Zhu, Zheng and Zheng, Wenzhao and Huang, Junjie and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2205.09743},
  year={2022}
}
```