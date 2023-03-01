<img src="..\figs\logo2.png" align="right" width="30%">

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


## BEVerse-Small

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4951 | 0.3512  | 0.6243 | 0.2694 | 0.3999 | 0.3292 | 0.1827 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.3364    | 0.1156    | 0.6753     | 0.3331     | 0.4460     | 0.4823     | 0.2772     |
|   Frame Lost   | 0.2485    | 0.0959    | 0.7413     | 0.4389     | 0.5898     | 0.8170     | 0.4445     |
|  Color Quant   | 0.2807    | 0.1630    | 0.8148     | 0.4651     | 0.6311     | 0.6511     | 0.4455     |
|  Motion Blur   | 0.2632    | 0.1455    | 0.7866     | 0.4399     | 0.5753     | 0.8424     | 0.4586     |
|   Brightness   | 0.3394    | 0.1935    | 0.7441     | 0.3736     | 0.4873     | 0.6357     | 0.3326     |
|   Low Light    | 0.1118    | 0.0373    | 0.9230     | 0.6900     | 0.8727     | 0.8600     | 0.7223     |
|      Fog       | 0.2849    | 0.1291    | 0.7858     | 0.4234     | 0.5105     | 0.6852     | 0.3921     |
|      Snow      | 0.0985    | 0.0357    | 0.9309     | 0.7389     | 0.8864     | 0.8695     | 0.7676     |


## Experiment Log

> Time: Fri Jan 27 18:04:25 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4041    | 0.1837    | 0.6471     | 0.2675     | 0.4118     | 0.3677     | 0.1834     |
|   Moderate   | 0.3134    | 0.0738    | 0.7142     | 0.3246     | 0.4234     | 0.5116     | 0.2615     |
|     Hard     | 0.2917    | 0.0891    | 0.6647     | 0.4071     | 0.5027     | 0.5676     | 0.3866     |
|   Average    | 0.3364    | 0.1156    | 0.6753     | 0.3331     | 0.4460     | 0.4823     | 0.2772     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4096    | 0.2165    | 0.6566     | 0.2711     | 0.4439     | 0.4313     | 0.1840     |
|   Moderate   | 0.2319    | 0.0624    | 0.7120     | 0.4094     | 0.5648     | 0.9071     | 0.4000     |
|     Hard     | 0.1042    | 0.0088    | 0.8554     | 0.6363     | 0.7607     | 1.1125     | 0.7494     |
|   Average    | 0.2485    | 0.0959    | 0.7413     | 0.4389     | 0.5898     | 0.8170     | 0.4445     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4553    | 0.3007    | 0.6679     | 0.2720     | 0.4570     | 0.3545     | 0.1994     |
|   Moderate   | 0.3010    | 0.1606    | 0.7969     | 0.4120     | 0.5427     | 0.6356     | 0.4059     |
|     Hard     | 0.0860    | 0.0277    | 0.9795     | 0.7113     | 0.8935     | 0.9633     | 0.7313     |
|   Average    | 0.2807    | 0.1630    | 0.8148     | 0.4651     | 0.6311     | 0.6511     | 0.4455     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4244    | 0.2684    | 0.6674     | 0.2734     | 0.4619     | 0.5055     | 0.1901     |
|   Moderate   | 0.2077    | 0.1050    | 0.8160     | 0.4829     | 0.6043     | 1.0073     | 0.5448     |
|     Hard     | 0.1576    | 0.0632    | 0.8763     | 0.5633     | 0.6596     | 1.0143     | 0.6409     |
|   Average    | 0.2632    | 0.1455    | 0.7866     | 0.4399     | 0.5753     | 0.8424     | 0.4586     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4419    | 0.2835    | 0.6642     | 0.2803     | 0.4229     | 0.4438     | 0.1874     |
|   Moderate   | 0.3142    | 0.1806    | 0.7401     | 0.4169     | 0.4958     | 0.7084     | 0.3995     |
|     Hard     | 0.2621    | 0.1164    | 0.8280     | 0.4235     | 0.5432     | 0.7550     | 0.4110     |
|   Average    | 0.3394    | 0.1935    | 0.7441     | 0.3736     | 0.4873     | 0.6357     | 0.3326     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1889    | 0.0616    | 0.8687     | 0.4987     | 0.7372     | 0.7860     | 0.5287     |
|   Moderate   | 0.0818    | 0.0357    | 0.9454     | 0.7853     | 0.9296     | 0.8863     | 0.8138     |
|     Hard     | 0.0648    | 0.0145    | 0.9549     | 0.7859     | 0.9514     | 0.9078     | 0.8245     |
|   Average    | 0.1118    | 0.0373    | 0.9230     | 0.6900     | 0.8727     | 0.8600     | 0.7223     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3424    | 0.2075    | 0.7236     | 0.4103     | 0.4876     | 0.6035     | 0.3882     |
|   Moderate   | 0.3089    | 0.1645    | 0.7598     | 0.4173     | 0.5103     | 0.6542     | 0.3919     |
|     Hard     | 0.2849    | 0.1291    | 0.7858     | 0.4234     | 0.5105     | 0.6852     | 0.3921     |
|   Average    | 0.3121    | 0.1671    | 0.7564     | 0.4170     | 0.5028     | 0.6476     | 0.3907     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1996    | 0.0648    | 0.8703     | 0.5050     | 0.7718     | 0.6847     | 0.4961     |
|   Moderate   | 0.0464    | 0.0198    | 0.9622     | 0.8566     | 0.9473     | 0.9669     | 0.9023     |
|     Hard     | 0.0496    | 0.0226    | 0.9602     | 0.8550     | 0.9402     | 0.9569     | 0.9045     |
|   Average    | 0.0985    | 0.0357    | 0.9309     | 0.7389     | 0.8864     | 0.8695     | 0.7676     |



## References

```bib
@article{zhang2022beverse,
  title={Beverse: Unified perception and prediction in birds-eye-view for vision-centric autonomous driving},
  author={Zhang, Yunpeng and Zhu, Zheng and Zheng, Wenzhao and Huang, Junjie and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2205.09743},
  year={2022}
}
```