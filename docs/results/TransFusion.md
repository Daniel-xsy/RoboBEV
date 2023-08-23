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


## TransFusion

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.6887 | 0.6453 | 0.2995 | 0.2552 | 0.3209 | 0.2765 | 0.1877 |
| |
| Cam Crash      | 0.6843    | 0.6368    | 0.3003     | 0.2569     | 0.3160     | 0.2808     | 0.1874     |
| Frame Lost     | 0.6447    | 0.5712    | 0.3187     | 0.2676     | 0.3393     | 0.2984     | 0.1855     |
| Color Quant    | 0.6819    | 0.6311    | 0.3005     | 0.2575     | 0.3151     | 0.2758     | 0.1873     |
| Motion Blur    | 0.6749    | 0.6195    | 0.3014     | 0.2586     | 0.3198     | 0.2831     | 0.1861     |
| Brightness     | 0.6843    | 0.6368    | 0.3003     | 0.2569     | 0.3160     | 0.2808     | 0.1874     |
| Low Light      | 0.6663    | 0.6044    | 0.3049     | 0.2613     | 0.3166     | 0.2855     | 0.1903     |
| Fog            | - | - | - | - | - | - | - |
| Snow           | - | - | - | - | - | - | - |


## Experiment Log

### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6862    | 0.6409    | 0.3006     | 0.2560     | 0.3187     | 0.2801     | 0.1874     |
| Moderate     | 0.6840    | 0.6363    | 0.3001     | 0.2569     | 0.3152     | 0.2812     | 0.1876     |
| Hard         | 0.6826    | 0.6332    | 0.3001     | 0.2578     | 0.3140     | 0.2810     | 0.1872     |
| Average      | 0.6843    | 0.6368    | 0.3003     | 0.2569     | 0.3160     | 0.2808     | 0.1874     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6597    | 0.5957    | 0.3140     | 0.2611     | 0.3369     | 0.2859     | 0.1834     |
| Moderate     | 0.6375    | 0.5584    | 0.3214     | 0.2695     | 0.3378     | 0.3033     | 0.1848     |
| Hard         | 0.6368    | 0.5595    | 0.3207     | 0.2721     | 0.3431     | 0.3059     | 0.1883     |
| Average      | 0.6447    | 0.5712    | 0.3187     | 0.2676     | 0.3393     | 0.2984     | 0.1855     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6879    | 0.6428    | 0.2989     | 0.2553     | 0.3157     | 0.2761     | 0.1884     |
| Moderate     | 0.6846    | 0.6359    | 0.2998     | 0.2574     | 0.3129     | 0.2759     | 0.1873     |
| Hard         | 0.6732    | 0.6146    | 0.3029     | 0.2599     | 0.3168     | 0.2754     | 0.1862     |
| Average      | 0.6819    | 0.6311    | 0.3005     | 0.2575     | 0.3151     | 0.2758     | 0.1873     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6841    | 0.6366    | 0.3000     | 0.2564     | 0.3202     | 0.2780     | 0.1869     |
| Moderate     | 0.6725    | 0.6152    | 0.3015     | 0.2594     | 0.3197     | 0.2844     | 0.1859     |
| Hard         | 0.6679    | 0.6068    | 0.3026     | 0.2600     | 0.3195     | 0.2868     | 0.1856     |
| Average      | 0.6749    | 0.6195    | 0.3014     | 0.2586     | 0.3198     | 0.2831     | 0.1861     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6862    | 0.6409    | 0.3006     | 0.2560     | 0.3187     | 0.2801     | 0.1874     |
| Moderate     | 0.6840    | 0.6363    | 0.3001     | 0.2569     | 0.3152     | 0.2812     | 0.1876     |
| Hard         | 0.6826    | 0.6332    | 0.3001     | 0.2578     | 0.3140     | 0.2810     | 0.1872     |
| Average      | 0.6843    | 0.6368    | 0.3003     | 0.2569     | 0.3160     | 0.2808     | 0.1874     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6713    | 0.6136    | 0.3053     | 0.2602     | 0.3158     | 0.2827     | 0.1912     |
| Moderate     | 0.6668    | 0.6050    | 0.3042     | 0.2609     | 0.3172     | 0.2848     | 0.1902     |
| Hard         | 0.6610    | 0.5946    | 0.3052     | 0.2627     | 0.3167     | 0.2890     | 0.1896     |
| Average      | 0.6663    | 0.6044    | 0.3049     | 0.2613     | 0.3166     | 0.2855     | 0.1903     |


## References

```bib
@inproceedings{bai2022transfusion,
  title={Transfusion: Robust lidar-camera fusion for 3d object detection with transformers},
  author={Bai, Xuyang and Hu, Zeyu and Zhu, Xinge and Huang, Qingqiu and Chen, Yilun and Fu, Hongbo and Tai, Chiew-Lan},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={1090--1099},
  year={2022}
}
```
