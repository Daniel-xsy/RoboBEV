<img src="..\figs\logo2.png" align="right" width="30%">

# RoboBEV Benchmark

The official [nuScenes metrics](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any) are considered in our benchmark:

### Average Precision (AP)

The average precision (AP) defines a match by thresholding the 2D center distance d on the ground plane instead of the intersection over union (IoU). This is done in order to decouple detection from object size and orientation but also because objects with small footprints, like pedestrians and bikes, if detected with a small translation error, give $0$ IoU.
We then calculate AP as the normalized area under the precision-recall curve for recall and precision over 10%. Operating points where recall or precision is less than $10$% are removed in order to minimize the impact of noise commonly seen in low precision and recall regions. If no operating point in this region is achieved, the AP for that class is set to zero. We then average over-matching thresholds of $\mathbb{D}=\{0.5, 1, 2, 4\}$ meters and the set of classes $\mathbb{C}$ :

$$
\text{mAP}= \frac{1}{|\mathbb{C}||\mathbb{D}|}\sum_{c\in\mathbb{C}}\sum_{d\in\mathbb{D}}\text{AP}_{c,d} .
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
\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))] .
$$


## SRCN3D-Vov

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4746 | 0.3960 | 0.7375 | 0.2939 | 0.2773 | 0.7281 | 0.1974 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.3288    | 0.1456    | 0.7957     | 0.3079     | 0.3359     | 0.7921     | 0.2080     |
|   Frame Lost   | 0.3038    | 0.1146    | 0.8182     | 0.3162     | 0.3923     | 0.7999     | 0.2087     |
|  Color Quant   | 0.3794    | 0.2729    | 0.8189     | 0.3058     | 0.3728     | 0.8479     | 0.2325     |
|  Motion Blur   | 0.2978    | 0.1835    | 0.8811     | 0.3261     | 0.5673     | 0.9092     | 0.2633     |
|   Brightness   | 0.4323    | 0.3374    | 0.7731     | 0.2995     | 0.3386     | 0.7621     | 0.1909     |
|   Low Light    | 0.2171    | 0.1098    | 0.8868     | 0.4286     | 0.6419     | 1.2864     | 0.4210     |
|      Fog       | 0.4201    | 0.3232    | 0.7712     | 0.2997     | 0.3468     | 0.8005     | 0.1970     |
|      Snow      | 0.2041    | 0.0915    | 0.9705     | 0.3885     | 0.7113     | 1.1448     | 0.3604     |


## Experiment Log

> Time: Mon Feb 20 13:20:08 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3820    | 0.2208    | 0.7680     | 0.3003     | 0.3006     | 0.7250     | 0.1903     |
|   Moderate   | 0.3032    | 0.1032    | 0.8249     | 0.3077     | 0.3388     | 0.7998     | 0.2125     |
|     Hard     | 0.3014    | 0.1129    | 0.7942     | 0.3156     | 0.3683     | 0.8516     | 0.2212     |
|   Average    | 0.3288    | 0.1456    | 0.7957     | 0.3079     | 0.3359     | 0.7921     | 0.2080     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3903    | 0.2450    | 0.7641     | 0.2993     | 0.2984     | 0.7671     | 0.1926     |
|   Moderate   | 0.2905    | 0.0828    | 0.8155     | 0.3142     | 0.3684     | 0.8023     | 0.2089     |
|     Hard     | 0.2305    | 0.0161    | 0.8749     | 0.3352     | 0.5102     | 0.8304     | 0.2247     |
|   Average    | 0.3038    | 0.1146    | 0.8182     | 0.3162     | 0.3923     | 0.7999     | 0.2087     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4611    | 0.3745    | 0.7464     | 0.2950     | 0.2960     | 0.7233     | 0.2003     |
|   Moderate   | 0.4012    | 0.2965    | 0.7987     | 0.2999     | 0.3575     | 0.7976     | 0.2169     |
|     Hard     | 0.2759    | 0.1478    | 0.9116     | 0.3226     | 0.4649     | 1.0227     | 0.2804     |
|   Average    | 0.3794    | 0.2729    | 0.8189     | 0.3058     | 0.3728     | 0.8479     | 0.2325     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4377    | 0.3457    | 0.7575     | 0.2959     | 0.3435     | 0.7532     | 0.2009     |
|   Moderate   | 0.2593    | 0.1316    | 0.8984     | 0.3288     | 0.6062     | 0.9508     | 0.2806     |
|     Hard     | 0.1964    | 0.0731    | 0.9874     | 0.3536     | 0.7523     | 1.0237     | 0.3085     |
|   Average    | 0.2978    | 0.1835    | 0.8811     | 0.3261     | 0.5673     | 0.9092     | 0.2633     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4655    | 0.3843    | 0.7431     | 0.2948     | 0.3008     | 0.7368     | 0.1905     |
|   Moderate   | 0.4326    | 0.3372    | 0.7704     | 0.2983     | 0.3399     | 0.7585     | 0.1929     |
|     Hard     | 0.3987    | 0.2907    | 0.8057     | 0.3053     | 0.3751     | 0.7911     | 0.1894     |
|   Average    | 0.4323    | 0.3374    | 0.7731     | 0.2995     | 0.3386     | 0.7621     | 0.1909     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3021    | 0.1864    | 0.8367     | 0.3314     | 0.4867     | 1.0317     | 0.2564     |
|   Moderate   | 0.2102    | 0.1053    | 0.8739     | 0.4546     | 0.6251     | 1.3182     | 0.4706     |
|     Hard     | 0.1389    | 0.0377    | 0.9498     | 0.4997     | 0.8139     | 1.5094     | 0.5361     |
|   Average    | 0.2171    | 0.1098    | 0.8868     | 0.4286     | 0.6419     | 1.2864     | 0.4210     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4367    | 0.3463    | 0.7599     | 0.2972     | 0.3260     | 0.7879     | 0.1938     |
|   Moderate   | 0.4207    | 0.3251    | 0.7707     | 0.2994     | 0.3488     | 0.8001     | 0.1998     |
|     Hard     | 0.4029    | 0.2982    | 0.7829     | 0.3024     | 0.3655     | 0.8134     | 0.1975     |
|   Average    | 0.4201    | 0.3232    | 0.7712     | 0.2997     | 0.3468     | 0.8005     | 0.1970     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2865    | 0.1693    | 0.8913     | 0.3338     | 0.5214     | 0.9755     | 0.2592     |
|   Moderate   | 0.1823    | 0.0634    | 1.0031     | 0.3727     | 0.7803     | 1.1562     | 0.3410     |
|     Hard     | 0.1436    | 0.0417    | 1.0171     | 0.4590     | 0.8321     | 1.3027     | 0.4810     |
|   Average    | 0.2041    | 0.0915    | 0.9705     | 0.3885     | 0.7113     | 1.1448     | 0.3604     |



## References

```bib
@article{shi2022srcn3d,
  title={Srcn3d: Sparse r-cnn 3d surround-view camera object detection and tracking for autonomous driving},
  author={Shi, Yining and Shen, Jingyan and Sun, Yifan and Wang, Yunlong and Li, Jiaxin and Sun, Shiqi and Jiang, Kun and Yang, Diange},
  journal={arXiv preprint arXiv:2206.14451},
  year={2022}
}
```