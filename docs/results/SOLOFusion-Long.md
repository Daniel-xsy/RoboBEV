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


## SOLOFusion-Long-Only

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.4850    | 0.3862    | 0.6292     | 0.2840     | 0.6387     | 0.3151     | 0.2141     |
| |
| Cam Crash      | 0.3159    | 0.1173    | 0.7462     | 0.2938     | 0.6939     | 0.4614     | 0.2327     |
| Frame Lost     | 0.2490    | 0.1121    | 0.7824     | 0.3529     | 0.8133     | 0.8167     | 0.3249     |
| Color Quant    | 0.3598    | 0.2233    | 0.7704     | 0.3206     | 0.7326     | 0.4266     | 0.2681     |
| Motion Blur    | 0.3460    | 0.1969    | 0.7765     | 0.2973     | 0.7849     | 0.4262     | 0.2395     |
| Brightness     | 0.4002    | 0.2726    | 0.7163     | 0.3113     | 0.6754     | 0.4251     | 0.2328     |
| Low Light      | 0.2814    | 0.1301    | 0.7669     | 0.3701     | 0.7913     | 0.5548     | 0.3534     |
| Fog            | 0.3991    | 0.2570    | 0.7230     | 0.2947     | 0.7084     | 0.3678     | 0.2002     |
| Snow           | 0.1480    | 0.0590    | 0.8901     | 0.5666     | 0.9179     | 0.7932     | 0.6480     |


## Experiment Log

> Time: Thu Apr  6 15:37:12 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3654    | 0.1912    | 0.7037     | 0.2866     | 0.6913     | 0.4032     | 0.2176     |
| Moderate     | 0.2961    | 0.0833    | 0.8047     | 0.2879     | 0.6726     | 0.4534     | 0.2368     |
| Hard         | 0.2861    | 0.0775    | 0.7301     | 0.3070     | 0.7178     | 0.5276     | 0.2438     |
| Average      | 0.3159    | 0.1173    | 0.7462     | 0.2938     | 0.6939     | 0.4614     | 0.2327     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3804    | 0.2524    | 0.6895     | 0.2951     | 0.7106     | 0.5281     | 0.2355     |
| Moderate     | 0.2239    | 0.0723    | 0.8009     | 0.3184     | 0.8774     | 0.8632     | 0.2625     |
| Hard         | 0.1427    | 0.0115    | 0.8568     | 0.4451     | 0.8519     | 1.0589     | 0.4766     |
| Average      | 0.2490    | 0.1121    | 0.7824     | 0.3529     | 0.8133     | 0.8167     | 0.3249     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4541    | 0.3436    | 0.6797     | 0.2846     | 0.6621     | 0.3287     | 0.2217     |
| Moderate     | 0.3806    | 0.2343    | 0.7579     | 0.2941     | 0.7034     | 0.3848     | 0.2249     |
| Hard         | 0.2446    | 0.0918    | 0.8736     | 0.3831     | 0.8324     | 0.5662     | 0.3578     |
| Average      | 0.3598    | 0.2233    | 0.7704     | 0.3206     | 0.7326     | 0.4266     | 0.2681     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4371    | 0.3207    | 0.6822     | 0.2876     | 0.7112     | 0.3401     | 0.2118     |
| Moderate     | 0.3227    | 0.1592    | 0.7891     | 0.2963     | 0.7985     | 0.4414     | 0.2432     |
| Hard         | 0.2783    | 0.1109    | 0.8581     | 0.3081     | 0.8450     | 0.4970     | 0.2635     |
| Average      | 0.3460    | 0.1969    | 0.7765     | 0.2973     | 0.7849     | 0.4262     | 0.2395     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4501    | 0.3321    | 0.6740     | 0.2890     | 0.6555     | 0.3415     | 0.1992     |
| Moderate     | 0.3998    | 0.2642    | 0.7303     | 0.2993     | 0.6977     | 0.3963     | 0.1992     |
| Hard         | 0.3507    | 0.2215    | 0.7446     | 0.3456     | 0.6731     | 0.5376     | 0.3001     |
| Average      | 0.4002    | 0.2726    | 0.7163     | 0.3113     | 0.6754     | 0.4251     | 0.2328     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3408    | 0.1831    | 0.7481     | 0.3042     | 0.7764     | 0.4396     | 0.2389     |
| Moderate     | 0.2859    | 0.1401    | 0.7622     | 0.3786     | 0.8201     | 0.5294     | 0.3511     |
| Hard         | 0.2176    | 0.0673    | 0.7903     | 0.4276     | 0.7773     | 0.6955     | 0.4701     |
| Average      | 0.2814    | 0.1301    | 0.7669     | 0.3701     | 0.7913     | 0.5548     | 0.3534     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4132    | 0.2766    | 0.7144     | 0.2942     | 0.6912     | 0.3548     | 0.1966     |
| Moderate     | 0.3985    | 0.2580    | 0.7213     | 0.2941     | 0.7152     | 0.3705     | 0.2042     |
| Hard         | 0.3856    | 0.2364    | 0.7334     | 0.2959     | 0.7188     | 0.3780     | 0.1997     |
| Average      | 0.3991    | 0.2570    | 0.7230     | 0.2947     | 0.7084     | 0.3678     | 0.2002     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2672    | 0.1224    | 0.7825     | 0.3796     | 0.8374     | 0.5530     | 0.3871     |
| Moderate     | 0.0931    | 0.0295    | 0.9408     | 0.6561     | 0.9475     | 0.8910     | 0.7811     |
| Hard         | 0.0835    | 0.0253    | 0.9469     | 0.6640     | 0.9687     | 0.9357     | 0.7758     |
| Average      | 0.1480    | 0.0590    | 0.8901     | 0.5666     | 0.9179     | 0.7932     | 0.6480     |



## References

```bib
@article{Park2022TimeWT,
  title={Time Will Tell: New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection},
  author={Park, Jinhyung and Xu, Chenfeng and Yang, Shijia and Keutzer, Kurt and Kitani, Kris and Tomizuka, Masayoshi and Zhan, Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
