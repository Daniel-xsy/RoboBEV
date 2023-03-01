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


## PolarFormer-Vov-DD3D-trainval-Pretrain

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.5616  | 0.5004  | 0.5826  | 0.2621 | 0.2473  | 0.6015  | 0.1926  |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.3904    | 0.1900    | 0.6509     | 0.2740     | 0.3012     | 0.6204     | 0.1996     |
|   Frame Lost   | 0.3593    | 0.1515    | 0.6861     | 0.2811     | 0.3437     | 0.6551     | 0.1982     |
|  Color Quant   | 0.4328    | 0.3386    | 0.7216     | 0.2732     | 0.3562     | 0.7960     | 0.2308     |
|  Motion Blur   | 0.3448    | 0.2196    | 0.7956     | 0.2894     | 0.5065     | 0.8103     | 0.2485     |
|   Brightness   | 0.5055    | 0.4292    | 0.6443     | 0.2659     | 0.3024     | 0.6856     | 0.1931     |
|   Low Light    | 0.2719    | 0.1597    | 0.8008     | 0.3773     | 0.5759     | 1.1300     | 0.3724     |
|      Fog       | 0.4905    | 0.4073    | 0.6597     | 0.2685     | 0.3026     | 0.7073     | 0.1934     |
|      Snow      | 0.2662    | 0.1438    | 0.8749     | 0.3150     | 0.5717     | 1.1031     | 0.3027     |


## Experiment Log

> Time: Fri Feb 17 21:58:00 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4490    | 0.2844    | 0.6244     | 0.2702     | 0.2638     | 0.5855     | 0.1874     |
|   Moderate   | 0.3621    | 0.1375    | 0.6635     | 0.2748     | 0.2933     | 0.6313     | 0.2039     |
|     Hard     | 0.3601    | 0.1482    | 0.6647     | 0.2771     | 0.3465     | 0.6443     | 0.2076     |
|   Average    | 0.3904    | 0.1900    | 0.6509     | 0.2740     | 0.3012     | 0.6204     | 0.1996     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4635    | 0.3162    | 0.6104     | 0.2660     | 0.2684     | 0.6137     | 0.1881     |
|   Moderate   | 0.3403    | 0.1126    | 0.6848     | 0.2781     | 0.3319     | 0.6669     | 0.1984     |
|     Hard     | 0.2742    | 0.0256    | 0.7630     | 0.2991     | 0.4309     | 0.6846     | 0.2081     |
|   Average    | 0.3593    | 0.1515    | 0.6861     | 0.2811     | 0.3437     | 0.6551     | 0.1982     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.5379    | 0.4711    | 0.6206     | 0.2618     | 0.2622     | 0.6363     | 0.1954     |
|   Moderate   | 0.4610    | 0.3650    | 0.7013     | 0.2684     | 0.3268     | 0.7144     | 0.2044     |
|     Hard     | 0.2994    | 0.1798    | 0.8430     | 0.2895     | 0.4797     | 1.0374     | 0.2926     |
|   Average    | 0.4328    | 0.3386    | 0.7216     | 0.2732     | 0.3562     | 0.7960     | 0.2308     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.5108    | 0.4268    | 0.6242     | 0.2631     | 0.2969     | 0.6500     | 0.1914     |
|   Moderate   | 0.2943    | 0.1527    | 0.8394     | 0.2929     | 0.5562     | 0.8669     | 0.2651     |
|     Hard     | 0.2292    | 0.0794    | 0.9233     | 0.3123     | 0.6663     | 0.9140     | 0.2889     |
|   Average    | 0.3448    | 0.2196    | 0.7956     | 0.2894     | 0.5065     | 0.8103     | 0.2485     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.5503    | 0.4857    | 0.5958     | 0.2619     | 0.2620     | 0.6128     | 0.1929     |
|   Moderate   | 0.5051    | 0.4285    | 0.6481     | 0.2653     | 0.3021     | 0.6842     | 0.1921     |
|     Hard     | 0.4611    | 0.3735    | 0.6890     | 0.2705     | 0.3430     | 0.7599     | 0.1942     |
|   Average    | 0.5055    | 0.4292    | 0.6443     | 0.2659     | 0.3024     | 0.6856     | 0.1931     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3784    | 0.2599    | 0.7206     | 0.2833     | 0.4051     | 0.8605     | 0.2464     |
|   Moderate   | 0.2639    | 0.1548    | 0.7919     | 0.3852     | 0.5945     | 1.1108     | 0.3627     |
|     Hard     | 0.1733    | 0.0645    | 0.8900     | 0.4634     | 0.7282     | 1.4186     | 0.5080     |
|   Average    | 0.2719    | 0.1597    | 0.8008     | 0.3773     | 0.5759     | 1.1300     | 0.3724     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.5146    | 0.4392    | 0.6368     | 0.2667     | 0.2891     | 0.6682     | 0.1888     |
|   Moderate   | 0.4908    | 0.4080    | 0.6566     | 0.2685     | 0.3037     | 0.7103     | 0.1927     |
|     Hard     | 0.4661    | 0.3748    | 0.6857     | 0.2703     | 0.3151     | 0.7433     | 0.1987     |
|   Average    | 0.4905    | 0.4073    | 0.6597     | 0.2685     | 0.3026     | 0.7073     | 0.1934     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3416    | 0.2290    | 0.7940     | 0.2871     | 0.4137     | 0.9780     | 0.2558     |
|   Moderate   | 0.2461    | 0.1191    | 0.8875     | 0.3180     | 0.6220     | 1.1479     | 0.3067     |
|     Hard     | 0.2108    | 0.0832    | 0.9433     | 0.3400     | 0.6793     | 1.1835     | 0.3455     |
|   Average    | 0.2662    | 0.1438    | 0.8749     | 0.3150     | 0.5717     | 1.1031     | 0.3027     |



## References

```bib
@article{jiang2022polarformer,
  title={Polarformer: Multi-camera 3d object detection with polar transformers},
  author={Jiang, Yanqin and Zhang, Li and Miao, Zhenwei and Zhu, Xiatian and Gao, Jin and Hu, Weiming and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2206.15398},
  year={2022}
}
```