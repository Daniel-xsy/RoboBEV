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


## BEVerse-Tiny-SingleFrame

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.1603 | 0.0826 | 0.8298 | 0.5296 | 0.8771 | 1.2639 | 0.5739 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.0639    | 0.0165    | 0.9135     | 0.7574     | 0.9522     | 1.1890     | 0.8201     |
|   Frame Lost   | 0.0508    | 0.0141    | 0.9455     | 0.8181     | 0.9221     | 1.1765     | 0.8765     |
|  Color Quant   | 0.0642    | 0.0317    | 0.9478     | 0.7735     | 0.9723     | 1.2508     | 0.8397     |
|  Motion Blur   | 0.0540    | 0.0230    | 0.9556     | 0.8028     | 0.9339     | 1.2137     | 0.8826     |
|   Brightness   | 0.0683    | 0.0360    | 0.9369     | 0.7315     | 0.9878     | 1.3048     | 0.8531     |
|   Low Light    | 0.0100    | 0.0005    | 1.0097     | 0.9474     | 1.0048     | 1.1073     | 0.9561     |
|      Fog       | 0.0402    | 0.0179    | 0.9789     | 0.8230     | 1.0094     | 1.3083     | 0.8962     |
|      Snow      | 0.0107    | 0.0017    | 1.0021     | 0.9468     | 0.9968     | 1.1652     | 0.9612     |


## Experiment Log

> Time: Fri Jan 27 17:36:37 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1105    | 0.0333    | 0.8594     | 0.6063     | 0.9176     | 1.2188     | 0.6788     |
|   Moderate   | 0.0443    | 0.0074    | 0.9342     | 0.8000     | 0.9992     | 1.2436     | 0.8607     |
|     Hard     | 0.0371    | 0.0088    | 0.9468     | 0.8659     | 0.9398     | 1.1047     | 0.9207     |
|   Average    | 0.0639    | 0.0165    | 0.9135     | 0.7574     | 0.9522     | 1.1890     | 0.8201     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0970    | 0.0368    | 0.8964     | 0.6687     | 0.8704     | 1.1819     | 0.7786     |
|   Moderate   | 0.0341    | 0.0052    | 0.9542     | 0.8665     | 0.9395     | 1.1732     | 0.9253     |
|     Hard     | 0.0213    | 0.0001    | 0.9860     | 0.9192     | 0.9565     | 1.1744     | 0.9255     |
|   Average    | 0.0508    | 0.0141    | 0.9455     | 0.8181     | 0.9221     | 1.1765     | 0.8765     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1301    | 0.0647    | 0.8621     | 0.5994     | 0.8774     | 1.1694     | 0.6839     |
|   Moderate   | 0.0481    | 0.0290    | 0.9708     | 0.8001     | 1.0287     | 1.2546     | 0.8931     |
|     Hard     | 0.0144    | 0.0014    | 1.0106     | 0.9209     | 1.0109     | 1.3283     | 0.9420     |
|   Average    | 0.0642    | 0.0317    | 0.9478     | 0.7735     | 0.9723     | 1.2508     | 0.8397     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1011    | 0.0513    | 0.9215     | 0.6710     | 0.8687     | 1.2238     | 0.7845     |
|   Moderate   | 0.0338    | 0.0123    | 0.9685     | 0.8677     | 0.9575     | 1.1947     | 0.9298     |
|     Hard     | 0.0272    | 0.0054    | 0.9768     | 0.8697     | 0.9754     | 1.2226     | 0.9336     |
|   Average    | 0.0540    | 0.0230    | 0.9556     | 0.8028     | 0.9339     | 1.2137     | 0.8826     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1041    | 0.0627    | 0.8875     | 0.6637     | 0.9502     | 1.2111     | 0.7710     |
|   Moderate   | 0.0610    | 0.0313    | 0.9490     | 0.7319     | 0.9766     | 1.3384     | 0.8890     |
|     Hard     | 0.0397    | 0.0139    | 0.9742     | 0.7989     | 1.0365     | 1.3648     | 0.8993     |
|   Average    | 0.0683    | 0.0360    | 0.9369     | 0.7315     | 0.9878     | 1.3048     | 0.8531     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0153    | 0.0011    | 1.0137     | 0.9208     | 0.9983     | 1.1611     | 0.9332     |
|   Moderate   | 0.0146    | 0.0005    | 1.0155     | 0.9214     | 1.0162     | 1.1608     | 0.9350     |
|     Hard     | 0.0000    | 0.0000    | 1.0000     | 1.0000     | 1.0000     | 1.0000     | 1.0000     |
|   Average    | 0.0100    | 0.0005    | 1.0097     | 0.9474     | 1.0048     | 1.1073     | 0.9561     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0491    | 0.0256    | 0.9654     | 0.7997     | 1.0261     | 1.2851     | 0.8720     |
|   Moderate   | 0.0436    | 0.0168    | 0.9751     | 0.8008     | 1.0317     | 1.3075     | 0.8725     |
|     Hard     | 0.0278    | 0.0114    | 0.9961     | 0.8684     | 0.9704     | 1.3323     | 0.9440     |
|   Average    | 0.0402    | 0.0179    | 0.9789     | 0.8230     | 1.0094     | 1.3083     | 0.8962     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0181    | 0.0045    | 0.9994     | 0.9203     | 0.9818     | 1.2245     | 0.9399     |
|   Moderate   | 0.0001    | 0.0001    | 1.0000     | 1.0000     | 1.0000     | 1.0000     | 1.0000     |
|     Hard     | 0.0139    | 0.0007    | 1.0068     | 0.9202     | 1.0087     | 1.2711     | 0.9436     |
|   Average    | 0.0107    | 0.0017    | 1.0021     | 0.9468     | 0.9968     | 1.1652     | 0.9612     |



## References

```bib
@article{zhang2022beverse,
  title={Beverse: Unified perception and prediction in birds-eye-view for vision-centric autonomous driving},
  author={Zhang, Yunpeng and Zhu, Zheng and Zheng, Wenzhao and Huang, Junjie and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2205.09743},
  year={2022}
}
```