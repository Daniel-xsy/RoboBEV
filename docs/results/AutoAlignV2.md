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


## AutoAlignV2

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.6139 | 0.5649 | 0.3300 | 0.2699 | 0.4226 | 0.4644 | 0.1983 |
| |
| Cam Crash      | 0.5849    | 0.5137    | 0.3393     | 0.2710     | 0.4329     | 0.4786     | 0.1977     |
| Frame Lost     | 0.5832    | 0.5104    | 0.3396     | 0.2718     | 0.4315     | 0.4787     | 0.1984     |
| Color Quant    | 0.6006    | 0.5405    | 0.3347     | 0.2703     | 0.4254     | 0.4708     | 0.1950     |
| Motion Blur    | 0.5901    | 0.5188    | 0.3336     | 0.2708     | 0.4243     | 0.4715     | 0.1923     |
| Brightness     | 0.6076    | 0.5497    | 0.3310     | 0.2699     | 0.4231     | 0.4560     | 0.1922     |
| Low Light      | 0.5770    | 0.4949    | 0.3390     | 0.2717     | 0.4157     | 0.4785     | 0.2000     |
| Fog            | - | - | - | - | - | - | - |
| Snow           | - | - | - | - | - | - | - |


## Experiment Log

### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.5879    | 0.5200    | 0.3382     | 0.2702     | 0.4392     | 0.4745     | 0.1983     |
| Moderate     | 0.5852    | 0.5144    | 0.3408     | 0.2710     | 0.4333     | 0.4778     | 0.1973     |
| Hard         | 0.5816    | 0.5068    | 0.3390     | 0.2718     | 0.4262     | 0.4835     | 0.1976     |
| Average      | 0.5849    | 0.5137    | 0.3393     | 0.2710     | 0.4329     | 0.4786     | 0.1977     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.5965    | 0.5341    | 0.3353     | 0.2707     | 0.4264     | 0.4745     | 0.1984     |
| Moderate     | 0.5803    | 0.5046    | 0.3408     | 0.2724     | 0.4294     | 0.4804     | 0.1969     |
| Hard         | 0.5728    | 0.4926    | 0.3426     | 0.2724     | 0.4387     | 0.4811     | 0.2000     |
| Average      | 0.5832    | 0.5104    | 0.3396     | 0.2718     | 0.4315     | 0.4787     | 0.1984     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6087    | 0.5569    | 0.3317     | 0.2699     | 0.4283     | 0.4699     | 0.1984     |
| Moderate     | 0.6040    | 0.5454    | 0.3336     | 0.2696     | 0.4225     | 0.4671     | 0.1940     |
| Hard         | 0.5892    | 0.5192    | 0.3389     | 0.2715     | 0.4255     | 0.4754     | 0.1926     |
| Average      | 0.6006    | 0.5405    | 0.3347     | 0.2703     | 0.4254     | 0.4708     | 0.1950     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6041    | 0.5446    | 0.3300     | 0.2698     | 0.4229     | 0.4667     | 0.1931     |
| Moderate     | 0.5867    | 0.5114    | 0.3347     | 0.2711     | 0.4219     | 0.4693     | 0.1926     |
| Hard         | 0.5796    | 0.5003    | 0.3362     | 0.2716     | 0.4281     | 0.4784     | 0.1911     |
| Average      | 0.5901    | 0.5188    | 0.3336     | 0.2708     | 0.4243     | 0.4715     | 0.1923     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6112    | 0.5569    | 0.3310     | 0.2700     | 0.4204     | 0.4587     | 0.1923     |
| Moderate     | 0.6073    | 0.5493    | 0.3310     | 0.2699     | 0.4244     | 0.4556     | 0.1926     |
| Hard         | 0.6043    | 0.5427    | 0.3309     | 0.2697     | 0.4245     | 0.4538     | 0.1916     |
| Average      | 0.6076    | 0.5497    | 0.3310     | 0.2699     | 0.4231     | 0.4560     | 0.1922     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.5904    | 0.5193    | 0.3344     | 0.2707     | 0.4147     | 0.4744     | 0.1983     |
| Moderate     | 0.5792    | 0.4983    | 0.3375     | 0.2714     | 0.4145     | 0.4756     | 0.2001     |
| Hard         | 0.5612    | 0.4671    | 0.3452     | 0.2731     | 0.4178     | 0.4854     | 0.2015     |
| Average      | 0.5770    | 0.4949    | 0.3390     | 0.2717     | 0.4157     | 0.4785     | 0.2000     |


## References

```bib
@article{chen2022autoalignv2,
  title={AutoAlignV2: Deformable Feature Aggregation for Dynamic Multi-Modal 3D Object Detection},
  author={Chen, Zehui and Li, Zhenyu and Zhang, Shiquan and Fang, Liangji and Jiang, Qinhong and Zhao, Feng},
  journal={ECCV},
  year={2022}
}
```
