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


## SRCN3D-R101

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4286 | 0.3373 | 0.7783 | 0.2873 | 0.3665 | 0.7806 | 0.1878 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.2947    | 0.1172    | 0.8369     | 0.3017     | 0.4403     | 0.8506     | 0.2097     |
|   Frame Lost   | 0.2681    | 0.0924    | 0.8637     | 0.3303     | 0.4798     | 0.8725     | 0.2349     |
|  Color Quant   | 0.3318    | 0.2199    | 0.8696     | 0.3041     | 0.4747     | 0.8877     | 0.2458     |
|  Motion Blur   | 0.2609    | 0.1361    | 0.9026     | 0.3524     | 0.5788     | 0.9964     | 0.2927     |
|   Brightness   | 0.4074    | 0.3133    | 0.7936     | 0.2911     | 0.3974     | 0.8227     | 0.1877     |
|   Low Light    | 0.2590    | 0.1406    | 0.8586     | 0.3642     | 0.5773     | 1.1257     | 0.3353     |
|      Fog       | 0.3940    | 0.2932    | 0.7993     | 0.2919     | 0.3978     | 0.8428     | 0.1944     |
|      Snow      | 0.1920    | 0.0734    | 0.9372     | 0.3996     | 0.7302     | 1.2366     | 0.3803     |


## Experiment Log

> Time: Mon Feb 20 13:17:16 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3399    | 0.1823    | 0.8094     | 0.2918     | 0.3864     | 0.8221     | 0.2024     |
|   Moderate   | 0.2633    | 0.0810    | 0.8743     | 0.3031     | 0.4362     | 0.9321     | 0.2260     |
|     Hard     | 0.2808    | 0.0884    | 0.8270     | 0.3101     | 0.4984     | 0.7976     | 0.2007     |
|   Average    | 0.2947    | 0.1172    | 0.8369     | 0.3017     | 0.4403     | 0.8506     | 0.2097     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3513    | 0.2016    | 0.8095     | 0.2907     | 0.3917     | 0.8144     | 0.1888     |
|   Moderate   | 0.2604    | 0.0643    | 0.8618     | 0.3110     | 0.4746     | 0.8643     | 0.2055     |
|     Hard     | 0.1925    | 0.0113    | 0.9199     | 0.3892     | 0.5732     | 0.9389     | 0.3103     |
|   Average    | 0.2681    | 0.0924    | 0.8637     | 0.3303     | 0.4798     | 0.8725     | 0.2349     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4088    | 0.3171    | 0.7977     | 0.2880     | 0.3850     | 0.8204     | 0.2064     |
|   Moderate   | 0.3458    | 0.2365    | 0.8506     | 0.2983     | 0.4660     | 0.8748     | 0.2347     |
|     Hard     | 0.2407    | 0.1060    | 0.9605     | 0.3260     | 0.5731     | 0.9678     | 0.2962     |
|   Average    | 0.3318    | 0.2199    | 0.8696     | 0.3041     | 0.4747     | 0.8877     | 0.2458     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3806    | 0.2754    | 0.8041     | 0.2932     | 0.4251     | 0.8358     | 0.2128     |
|   Moderate   | 0.2249    | 0.0861    | 0.9392     | 0.3454     | 0.6254     | 1.0136     | 0.2716     |
|     Hard     | 0.1772    | 0.0470    | 0.9646     | 0.4186     | 0.6858     | 1.1397     | 0.3938     |
|   Average    | 0.2609    | 0.1361    | 0.9026     | 0.3524     | 0.5788     | 0.9964     | 0.2927     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4250    | 0.3331    | 0.7815     | 0.2879     | 0.3760     | 0.7843     | 0.1864     |
|   Moderate   | 0.4079    | 0.3150    | 0.7942     | 0.2906     | 0.3965     | 0.8273     | 0.1875     |
|     Hard     | 0.3893    | 0.2916    | 0.8051     | 0.2947     | 0.4196     | 0.8565     | 0.1892     |
|   Average    | 0.4074    | 0.3133    | 0.7936     | 0.2911     | 0.3974     | 0.8227     | 0.1877     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3252    | 0.2111    | 0.8360     | 0.3017     | 0.4853     | 0.9329     | 0.2478     |
|   Moderate   | 0.2707    | 0.1455    | 0.8588     | 0.3236     | 0.5625     | 1.0326     | 0.2762     |
|     Hard     | 0.1812    | 0.0653    | 0.8809     | 0.4672     | 0.6840     | 1.4115     | 0.4820     |
|   Average    | 0.2590    | 0.1406    | 0.8586     | 0.3642     | 0.5773     | 1.1257     | 0.3353     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4081    | 0.3101    | 0.7893     | 0.2894     | 0.3859     | 0.8155     | 0.1892     |
|   Moderate   | 0.3976    | 0.2960    | 0.7980     | 0.2909     | 0.3906     | 0.8330     | 0.1919     |
|     Hard     | 0.3763    | 0.2736    | 0.8106     | 0.2953     | 0.4169     | 0.8800     | 0.2021     |
|   Average    | 0.3940    | 0.2932    | 0.7993     | 0.2919     | 0.3978     | 0.8428     | 0.1944     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2503    | 0.1200    | 0.9046     | 0.3344     | 0.5726     | 1.1201     | 0.2858     |
|   Moderate   | 0.1706    | 0.0552    | 0.9387     | 0.4265     | 0.7846     | 1.2776     | 0.4201     |
|     Hard     | 0.1550    | 0.0450    | 0.9682     | 0.4380     | 0.8334     | 1.3122     | 0.4349     |
|   Average    | 0.1920    | 0.0734    | 0.9372     | 0.3996     | 0.7302     | 1.2366     | 0.3803     |



## References

```bib
@article{shi2022srcn3d,
  title={Srcn3d: Sparse r-cnn 3d surround-view camera object detection and tracking for autonomous driving},
  author={Shi, Yining and Shen, Jingyan and Sun, Yifan and Wang, Yunlong and Li, Jiaxin and Sun, Shiqi and Jiang, Kun and Yang, Diange},
  journal={arXiv preprint arXiv:2206.14451},
  year={2022}
}
```