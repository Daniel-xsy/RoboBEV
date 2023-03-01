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


## BEVDepth-r50

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4058 | 0.3328 | 0.6633 | 0.2714 | 0.5581 | 0.8763 | 0.2369 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.2638    | 0.1111    | 0.7407     | 0.2959     | 0.6373     | 1.0079     | 0.2749     |
|   Frame Lost   | 0.2141    | 0.0876    | 0.7890     | 0.4134     | 0.6728     | 1.0536     | 0.4498     |
|  Color Quant   | 0.2751    | 0.1865    | 0.8190     | 0.3292     | 0.6946     | 1.2008     | 0.3552     |
|  Motion Blur   | 0.2513    | 0.1508    | 0.8320     | 0.3516     | 0.7135     | 1.1084     | 0.3765     |
|   Brightness   | 0.2879    | 0.2090    | 0.7520     | 0.3646     | 0.6724     | 1.2089     | 0.3766     |
|   Low Light    | 0.1757    | 0.0820    | 0.8540     | 0.4509     | 0.8073     | 1.3149     | 0.5410     |
|      Fog       | 0.2903    | 0.1973    | 0.7900     | 0.3021     | 0.6973     | 1.0640     | 0.2940     |
|      Snow      | 0.0863    | 0.0350    | 0.9529     | 0.6682     | 0.9107     | 1.2750     | 0.7802     |


## Experiment Log

> Time: Mon Feb 13 17:09:27 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3132    | 0.1771    | 0.7131     | 0.2750     | 0.6217     | 0.9053     | 0.2379     |
|   Moderate   | 0.2282    | 0.0759    | 0.7800     | 0.3319     | 0.6699     | 1.0951     | 0.3160     |
|     Hard     | 0.2500    | 0.0802    | 0.7289     | 0.2807     | 0.6204     | 1.0232     | 0.2709     |
|   Average    | 0.2638    | 0.1111    | 0.7407     | 0.2959     | 0.6373     | 1.0079     | 0.2749     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3251    | 0.1962    | 0.7049     | 0.2735     | 0.5954     | 0.9158     | 0.2398     |
|   Moderate   | 0.2209    | 0.0576    | 0.7749     | 0.3321     | 0.6382     | 1.1222     | 0.3333     |
|     Hard     | 0.0962    | 0.0090    | 0.8872     | 0.6345     | 0.7849     | 1.1228     | 0.7764     |
|   Average    | 0.2141    | 0.0876    | 0.7890     | 0.4134     | 0.6728     | 1.0536     | 0.4498     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3748    | 0.2986    | 0.6970     | 0.2704     | 0.5783     | 0.9511     | 0.2475     |
|   Moderate   | 0.2981    | 0.1960    | 0.8174     | 0.2824     | 0.6306     | 1.1438     | 0.2688     |
|     Hard     | 0.1523    | 0.0650    | 0.9426     | 0.4348     | 0.8748     | 1.5074     | 0.5494     |
|   Average    | 0.2751    | 0.1865    | 0.8190     | 0.3292     | 0.6946     | 1.2008     | 0.3552     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3639    | 0.2705    | 0.7008     | 0.2739     | 0.6008     | 0.9018     | 0.2361     |
|   Moderate   | 0.2387    | 0.1139    | 0.8566     | 0.2890     | 0.7532     | 1.0651     | 0.2836     |
|     Hard     | 0.1513    | 0.0680    | 0.9385     | 0.4918     | 0.7865     | 1.3584     | 0.6099     |
|   Average    | 0.2513    | 0.1508    | 0.8320     | 0.3516     | 0.7135     | 1.1084     | 0.3765     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3568    | 0.2755    | 0.7028     | 0.2741     | 0.5896     | 1.0115     | 0.2428     |
|   Moderate   | 0.2706    | 0.1986    | 0.7529     | 0.4059     | 0.6913     | 1.2522     | 0.4370     |
|     Hard     | 0.2364    | 0.1529    | 0.8004     | 0.4137     | 0.7364     | 1.3629     | 0.4501     |
|   Average    | 0.2879    | 0.2090    | 0.7520     | 0.3646     | 0.6724     | 1.2089     | 0.3766     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2284    | 0.1263    | 0.8128     | 0.3639     | 0.7764     | 1.1842     | 0.3948     |
|   Moderate   | 0.1869    | 0.0848    | 0.8311     | 0.4170     | 0.8184     | 1.3781     | 0.4882     |
|     Hard     | 0.1118    | 0.0350    | 0.9182     | 0.5719     | 0.8270     | 1.3823     | 0.7400     |
|   Average    | 0.1757    | 0.0820    | 0.8540     | 0.4509     | 0.8073     | 1.3149     | 0.5410     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3205    | 0.2267    | 0.7648     | 0.2719     | 0.6404     | 1.0073     | 0.2507     |
|   Moderate   | 0.2980    | 0.1968    | 0.7906     | 0.2757     | 0.6809     | 1.0621     | 0.2563     |
|     Hard     | 0.2524    | 0.1685    | 0.8146     | 0.3586     | 0.7705     | 1.1227     | 0.3751     |
|   Average    | 0.2903    | 0.1973    | 0.7900     | 0.3021     | 0.6973     | 1.0640     | 0.2940     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1716    | 0.0709    | 0.8735     | 0.4202     | 0.8563     | 1.5095     | 0.4884     |
|   Moderate   | 0.0513    | 0.0238    | 0.9843     | 0.7900     | 0.9088     | 1.1445     | 0.9236     |
|     Hard     | 0.0362    | 0.0103    | 1.0010     | 0.7943     | 0.9669     | 1.1709     | 0.9287     |
|   Average    | 0.0863    | 0.0350    | 0.9529     | 0.6682     | 0.9107     | 1.2750     | 0.7802     |



## References

```bib
@article{li2022bevdepth,
  title={Bevdepth: Acquisition of reliable depth for multi-view 3d object detection},
  author={Li, Yinhao and Ge, Zheng and Yu, Guanyi and Yang, Jinrong and Wang, Zengran and Shi, Yukang and Sun, Jianjian and Li, Zeming},
  journal={arXiv preprint arXiv:2206.10092},
  year={2022}
}
```