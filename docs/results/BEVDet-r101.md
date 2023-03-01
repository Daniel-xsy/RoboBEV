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


## BEVDet-r101

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.3877 | 0.3008 | 0.7035 | 0.2752 | 0.5384 | 0.8715 | 0.2379 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.2622    | 0.1042    | 0.7821     | 0.3004     | 0.6028     | 0.9783     | 0.2715     |
|   Frame Lost   | 0.2065    | 0.0805    | 0.8248     | 0.4175     | 0.6754     | 1.0578     | 0.4474     |
|  Color Quant   | 0.2546    | 0.1566    | 0.8457     | 0.3361     | 0.6966     | 1.1529     | 0.3716     |
|  Motion Blur   | 0.2265    | 0.1278    | 0.8596     | 0.3785     | 0.7112     | 1.1344     | 0.4246     |
|   Brightness   | 0.2554    | 0.1738    | 0.8094     | 0.3770     | 0.7228     | 1.3752     | 0.4060     |
|   Low Light    | 0.1118    | 0.0426    | 0.9659     | 0.5550     | 0.8904     | 1.3003     | 0.6836     |
|      Fog       | 0.2495    | 0.1412    | 0.8460     | 0.3269     | 0.7007     | 1.1480     | 0.3376     |
|      Snow      | 0.0810    | 0.0296    | 0.9727     | 0.6758     | 0.9027     | 1.1803     | 0.7869     |


## Experiment Log

> Time: Tue Feb 28 20:42:32 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3082    | 0.1660    | 0.7519     | 0.2753     | 0.5733     | 0.8972     | 0.2503     |
|   Moderate   | 0.2295    | 0.0736    | 0.8124     | 0.3360     | 0.6021     | 1.1090     | 0.3223     |
|     Hard     | 0.2490    | 0.0731    | 0.7820     | 0.2899     | 0.6331     | 0.9288     | 0.2420     |
|   Average    | 0.2622    | 0.1042    | 0.7821     | 0.3004     | 0.6028     | 0.9783     | 0.2715     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3153    | 0.1793    | 0.7470     | 0.2761     | 0.5691     | 0.9157     | 0.2356     |
|   Moderate   | 0.2121    | 0.0530    | 0.8160     | 0.3389     | 0.6614     | 1.1060     | 0.3272     |
|     Hard     | 0.0922    | 0.0093    | 0.9113     | 0.6376     | 0.7958     | 1.1517     | 0.7795     |
|   Average    | 0.2065    | 0.0805    | 0.8248     | 0.4175     | 0.6754     | 1.0578     | 0.4474     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3533    | 0.2703    | 0.7292     | 0.2771     | 0.5894     | 0.9623     | 0.2605     |
|   Moderate   | 0.2681    | 0.1561    | 0.8443     | 0.2913     | 0.6634     | 1.1378     | 0.3003     |
|     Hard     | 0.1423    | 0.0435    | 0.9635     | 0.4400     | 0.8370     | 1.3586     | 0.5540     |
|   Average    | 0.2546    | 0.1566    | 0.8457     | 0.3361     | 0.6966     | 1.1529     | 0.3716     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3282    | 0.2388    | 0.7663     | 0.2800     | 0.6210     | 1.0018     | 0.2444     |
|   Moderate   | 0.2220    | 0.0908    | 0.8915     | 0.2927     | 0.7441     | 1.0967     | 0.3060     |
|     Hard     | 0.1293    | 0.0537    | 0.9211     | 0.5629     | 0.7684     | 1.3047     | 0.7233     |
|   Average    | 0.2265    | 0.1278    | 0.8596     | 0.3785     | 0.7112     | 1.1344     | 0.4246     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3312    | 0.2455    | 0.7542     | 0.2899     | 0.6251     | 1.0659     | 0.2464     |
|   Moderate   | 0.2380    | 0.1657    | 0.8096     | 0.4172     | 0.7460     | 1.4532     | 0.4756     |
|     Hard     | 0.1969    | 0.1101    | 0.8643     | 0.4239     | 0.7973     | 1.6066     | 0.4959     |
|   Average    | 0.2554    | 0.1738    | 0.8094     | 0.3770     | 0.7228     | 1.3752     | 0.4060     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1610    | 0.0677    | 0.9284     | 0.4312     | 0.8561     | 1.3176     | 0.5126     |
|   Moderate   | 0.1235    | 0.0421    | 0.9692     | 0.5079     | 0.8575     | 1.4856     | 0.6406     |
|     Hard     | 0.0508    | 0.0178    | 1.0001     | 0.7258     | 0.9576     | 1.0978     | 0.8977     |
|   Average    | 0.1118    | 0.0426    | 0.9659     | 0.5550     | 0.8904     | 1.3003     | 0.6836     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2875    | 0.1819    | 0.8215     | 0.2768     | 0.6697     | 1.0680     | 0.2667     |
|   Moderate   | 0.2570    | 0.1368    | 0.8542     | 0.2836     | 0.6997     | 1.1161     | 0.2759     |
|     Hard     | 0.2039    | 0.1050    | 0.8624     | 0.4203     | 0.7326     | 1.2599     | 0.4702     |
|   Average    | 0.2495    | 0.1412    | 0.8460     | 0.3269     | 0.7007     | 1.1480     | 0.3376     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1332    | 0.0500    | 0.9339     | 0.5057     | 0.9026     | 1.2062     | 0.5758     |
|   Moderate   | 0.0680    | 0.0238    | 0.9870     | 0.7211     | 0.8795     | 1.1984     | 0.8515     |
|     Hard     | 0.0418    | 0.0151    | 0.9972     | 0.8006     | 0.9261     | 1.1364     | 0.9334     |
|   Average    | 0.0810    | 0.0296    | 0.9727     | 0.6758     | 0.9027     | 1.1803     | 0.7869     |



## References

```bib
@article{huang2021bevdet,
  title={Bevdet: High-performance multi-camera 3d object detection in bird-eye-view},
  author={Huang, Junjie and Huang, Guan and Zhu, Zheng and Du, Dalong},
  journal={arXiv preprint arXiv:2112.11790},
  year={2021}
}
}
```