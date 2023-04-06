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


## BEVFusion Camera

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.4122 | 0.3556 | 0.6677 | 0.2727 | 0.5612 | 0.8954 | 0.2593 |
| |
| Cam Crash      | 0.2777    | 0.1232    | 0.7343     | 0.2745     | 0.6202     | 0.9324     | 0.2846     |
| Frame Lost     | 0.2255    | 0.0968    | 0.7888     | 0.3835     | 0.6476     | 1.0360     | 0.4393     |
| Color Quant    | 0.2763    | 0.1896    | 0.8245     | 0.3308     | 0.6832     | 1.1253     | 0.3633     |
| Motion Blur    | 0.2788    | 0.1895    | 0.7902     | 0.3204     | 0.7041     | 1.0588     | 0.3635     |
| Brightness     | 0.2902    | 0.2158    | 0.7857     | 0.3390     | 0.7030     | 1.2076     | 0.3497     |
| Low Light      | 0.1076    | 0.0422    | 0.9707     | 0.5462     | 0.9279     | 1.2293     | 0.6904     |
| Fog            |  |  |  |  |  |  |  |
| Snow           |  |  |  |  |  |  |  |


## Experiment Log

> Time: Thu Apr  6 10:06:26 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3137    | 0.1908    | 0.7159     | 0.2716     | 0.6189     | 0.9407     | 0.2697     |
| Moderate     | 0.2509    | 0.0859    | 0.7500     | 0.2747     | 0.6010     | 1.0205     | 0.2944     |
| Hard         | 0.2684    | 0.0930    | 0.7370     | 0.2772     | 0.6407     | 0.8359     | 0.2897     |
| Average      | 0.2777    | 0.1232    | 0.7343     | 0.2745     | 0.6202     | 0.9324     | 0.2846     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3308    | 0.2128    | 0.7183     | 0.2716     | 0.5973     | 0.9089     | 0.2598     |
| Moderate     | 0.2238    | 0.0662    | 0.7833     | 0.3263     | 0.6317     | 1.0767     | 0.3520     |
| Hard         | 0.1220    | 0.0115    | 0.8649     | 0.5526     | 0.7137     | 1.1223     | 0.7061     |
| Average      | 0.2255    | 0.0968    | 0.7888     | 0.3835     | 0.6476     | 1.0360     | 0.4393     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3821    | 0.3165    | 0.6928     | 0.2748     | 0.5747     | 0.9495     | 0.2702     |
| Moderate     | 0.2990    | 0.1993    | 0.7945     | 0.2857     | 0.6366     | 1.0470     | 0.2890     |
| Hard         | 0.1478    | 0.0531    | 0.9861     | 0.4318     | 0.8384     | 1.3793     | 0.5306     |
| Average      | 0.2763    | 0.1896    | 0.8245     | 0.3308     | 0.6832     | 1.1253     | 0.3633     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3707    | 0.3018    | 0.7237     | 0.2748     | 0.5953     | 0.9422     | 0.2657     |
| Moderate     | 0.2695    | 0.1631    | 0.8029     | 0.2810     | 0.7244     | 1.0163     | 0.3125     |
| Hard         | 0.1963    | 0.1035    | 0.8441     | 0.4055     | 0.7927     | 1.2178     | 0.5122     |
| Average      | 0.2788    | 0.1895    | 0.7902     | 0.3204     | 0.7041     | 1.0588     | 0.3635     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3579    | 0.2925    | 0.7268     | 0.2816     | 0.5994     | 1.0685     | 0.2754     |
| Moderate     | 0.2733    | 0.2011    | 0.7903     | 0.3640     | 0.7303     | 1.2392     | 0.3881     |
| Hard         | 0.2392    | 0.1537    | 0.8400     | 0.3715     | 0.7793     | 1.3150     | 0.3855     |
| Average      | 0.2902    | 0.2158    | 0.7857     | 0.3390     | 0.7030     | 1.2076     | 0.3497     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.1552    | 0.0645    | 0.9514     | 0.4168     | 0.8588     | 1.4046     | 0.5438     |
| Moderate     | 0.1178    | 0.0433    | 0.9656     | 0.5019     | 0.9491     | 1.1622     | 0.6220     |
| Hard         | 0.0497    | 0.0186    | 0.9951     | 0.7199     | 0.9757     | 1.1210     | 0.9055     |
| Average      | 0.1076    | 0.0422    | 0.9707     | 0.5462     | 0.9279     | 1.2293     | 0.6904     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         |  |  |  |  |  |  |  |
| Moderate     |  |  |  |  |  |  |  |
| Hard         |  |  |  |  |  |  |  |
| Average      |  |  |  |  |  |  |  |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         |  |  |  |  |  |  |  |
| Moderate     |  |  |  |  |  |  |  |
| Hard         |  |  |  |  |  |  |  |
| Average      |  |  |  |  |  |  |  |



## References

```bib
@inproceedings{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```
