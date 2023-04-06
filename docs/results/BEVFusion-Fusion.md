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


## BEVFusion Camera + LiDAR

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.7138 | 0.6852 | 0.2874 | 0.2539 | 0.3044 | 0.2554 | 0.1874 |
| |
| Cam Crash      | 0.6963    | 0.6525    | 0.2916     | 0.2565     | 0.3069     | 0.2603     | 0.1847     |
| Frame Lost     | 0.6931    | 0.6478    | 0.2927     | 0.2573     | 0.3112     | 0.2628     | 0.1840     |
| Color Quant    | 0.7044    | 0.6665    | 0.2886     | 0.2553     | 0.2980     | 0.2569     | 0.1891     |
| Motion Blur    | 0.6977    | 0.6557    | 0.2885     | 0.2560     | 0.3116     | 0.2598     | 0.1853     |
| Brightness     | 0.7018    | 0.6622    | 0.2926     | 0.2566     | 0.2996     | 0.2593     | 0.1852     |
| Low Light      | 0.6787    | 0.6210    | 0.2939     | 0.2600     | 0.2998     | 0.2739     | 0.1898     |
| Fog            | - | - | - | - | - | - | - |
| Snow           | - | - | - | - | - | - | - |


## Experiment Log

> Time: Thu Apr  6 11:46:46 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.7044    | 0.6667    | 0.2883     | 0.2560     | 0.3040     | 0.2570     | 0.1841     |
| Moderate     | 0.6946    | 0.6511    | 0.2945     | 0.2567     | 0.3082     | 0.2644     | 0.1855     |
| Hard         | 0.6898    | 0.6398    | 0.2920     | 0.2569     | 0.3086     | 0.2594     | 0.1846     |
| Average      | 0.6963    | 0.6525    | 0.2916     | 0.2565     | 0.3069     | 0.2603     | 0.1847     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.7027    | 0.6643    | 0.2904     | 0.2557     | 0.3049     | 0.2580     | 0.1852     |
| Moderate     | 0.6903    | 0.6438    | 0.2946     | 0.2575     | 0.3141     | 0.2659     | 0.1835     |
| Hard         | 0.6862    | 0.6354    | 0.2932     | 0.2587     | 0.3146     | 0.2646     | 0.1832     |
| Average      | 0.6931    | 0.6478    | 0.2927     | 0.2573     | 0.3112     | 0.2628     | 0.1840     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.7122    | 0.6818    | 0.2876     | 0.2544     | 0.3000     | 0.2557     | 0.1889     |
| Moderate     | 0.7076    | 0.6714    | 0.2889     | 0.2556     | 0.2925     | 0.2560     | 0.1880     |
| Hard         | 0.6934    | 0.6461    | 0.2894     | 0.2558     | 0.3016     | 0.2591     | 0.1904     |
| Average      | 0.7044    | 0.6665    | 0.2886     | 0.2553     | 0.2980     | 0.2569     | 0.1891     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.7091    | 0.6765    | 0.2867     | 0.2545     | 0.3117     | 0.2544     | 0.1840     |
| Moderate     | 0.6952    | 0.6500    | 0.2881     | 0.2563     | 0.3083     | 0.2594     | 0.1858     |
| Hard         | 0.6889    | 0.6406    | 0.2906     | 0.2572     | 0.3148     | 0.2655     | 0.1860     |
| Average      | 0.6977    | 0.6557    | 0.2885     | 0.2560     | 0.3116     | 0.2598     | 0.1853     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.7082    | 0.6760    | 0.2918     | 0.2556     | 0.3040     | 0.2593     | 0.1869     |
| Moderate     | 0.7000    | 0.6585    | 0.2932     | 0.2567     | 0.2989     | 0.2596     | 0.1840     |
| Hard         | 0.6971    | 0.6521    | 0.2927     | 0.2576     | 0.2958     | 0.2590     | 0.1846     |
| Average      | 0.7018    | 0.6622    | 0.2926     | 0.2566     | 0.2996     | 0.2593     | 0.1852     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.6875    | 0.6360    | 0.2923     | 0.2589     | 0.2989     | 0.2647     | 0.1897     |
| Moderate     | 0.6805    | 0.6232    | 0.2925     | 0.2594     | 0.2972     | 0.2715     | 0.1897     |
| Hard         | 0.6682    | 0.6038    | 0.2970     | 0.2617     | 0.3033     | 0.2855     | 0.1900     |
| Average      | 0.6787    | 0.6210    | 0.2939     | 0.2600     | 0.2998     | 0.2739     | 0.1898     |


## References

```bib
@inproceedings{liu2022bevfusion,
  title={BEVFusion: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation},
  author={Liu, Zhijian and Tang, Haotian and Amini, Alexander and Yang, Xingyu and Mao, Huizi and Rus, Daniela and Han, Song},
  booktitle={IEEE International Conference on Robotics and Automation (ICRA)},
  year={2023}
}
```
