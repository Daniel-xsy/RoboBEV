<img src="../figs/logo2.png" align="right" width="30%">

# RoboBEV Benchmark

The official [nuScenes metric](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any) are considered in our benchmark:

### Average Precision (AP)

The average precision (AP) defines a match by thresholding the 2D center distance d on the ground plane instead of the intersection over union (IoU). This is done in order to decouple detection from object size and orientation but also because objects with small footprints, like pedestrians and bikes, if detected with a small translation error, give 0 IoU.
We then calculate AP as the normalized area under the precision-recall curve for recall and precision over 10%. Operating points where recall or precision is less than 10% are removed in order to minimize the impact of noise commonly seen in low precision and recall regions. If no operating point in this region is achieved, the AP for that class is set to zero. We then average over-matching thresholds of $\mathbb{D}=\{0.5, 1, 2, 4\}$ meters and the set of classes $\mathbb{C}$ :

$$
\text{mAP}= \frac{1}{|\mathbb{C}||\mathbb{D}|}\sum_{c\in\mathbb{C}}\sum_{d\in\mathbb{D}}\text{AP}_{c,d}
$$

### True Positive (TP)

All TP metrics are calculated using $d=2$ m center distance during matching, and they are all designed to be positive scalars. Matching and scoring happen independently per class and each metric is the average of the cumulative mean at each achieved recall level above 10%. If a 10% recall is not achieved for a particular class, all TP errors for that class are set to 1. 

- **Average Translation Error (ATE)** is the Euclidean center distance in 2D (units in meters). 
- **Average Scale Error (ASE)** is the 3D intersection-over-union (IoU) after aligning orientation and translation (1 − IoU).
- **Average Orientation Error (AOE)** is the smallest yaw angle difference between prediction and ground truth (radians). All angles are measured on a full 360-degree period except for barriers where they are measured on a 180-degree period.
- **Average Velocity Error (AVE)** is the absolute velocity error as the L2 norm of the velocity differences in 2D (m/s).
- **Average Attribute Error (AAE)** is defined as 1 minus attribute classification accuracy (1 − acc).

### nuScenes Detection Score (NDS)
mAP with a threshold on IoU is perhaps the most popular metric for object detection. However, this metric can not capture all aspects of the nuScenes detection tasks, like velocity and attribute estimation. Further, it couples location, size, and orientation estimates. nuScenes proposed instead consolidating the different error types into a scalar score:

$$
\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))]
$$

## BEVFormer-Small-SingleFrame

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|||||||||
| Clean | 0.2622    | 0.1324    | 0.9352     | 0.3024     | 0.5556     | 1.1106     | 0.2466     |
|||||||||
| Camera Crash | 0.2013    | 0.0425    | 0.9844     | 0.3306     | 0.6330     | 1.0969     | 0.2556     |
| Frame Lost | 0.1638    | 0.0292    | 1.0051     | 0.4294     | 0.6963     | 1.1418     | 0.3954     |
| Color Quant | 0.2313    | 0.1041    | 0.9625     | 0.3131     | 0.6435     | 1.1686     | 0.2882     |
| Motion Blur | 0.1916    | 0.0676    | 0.9741     | 0.3644     | 0.7525     | 1.3062     | 0.3307     |
| Brightness | 0.2520    | 0.1250    | 0.9484     | 0.3034     | 0.6046     | 1.1318     | 0.2486     |
| Low Light | 0.1868    | 0.0624    | 0.9414     | 0.3984     | 0.7185     | 1.3064     | 0.3859     |
| Fog | 0.2442    | 0.1181    | 0.9498     | 0.3055     | 0.6343     | 1.1806     | 0.2592     |
| Snow | 0.1414    | 0.0294    | 1.0231     | 0.4242     | 0.8644     | 1.3622     | 0.4444     |

## Experiment Log

>Time: Tue Feb 14 09:35:41 2023

### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2283    | 0.0744    | 0.9497     | 0.3159     | 0.5703     | 1.0525     | 0.2526     |
|   Moderate   | 0.2006    | 0.0361    | 0.9911     | 0.3330     | 0.5784     | 1.1691     | 0.2721     |
|     Hard     | 0.1750    | 0.0171    | 1.0125     | 0.3429     | 0.7504     | 1.0690     | 0.2422     |
|   Average    | 0.2013    | 0.0425    | 0.9844     | 0.3306     | 0.6330     | 1.0969     | 0.2556     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2244    | 0.0706    | 0.9614     | 0.3148     | 0.5881     | 1.1012     | 0.2451     |
|   Moderate   | 0.1767    | 0.0156    | 1.0172     | 0.3467     | 0.7087     | 1.1011     | 0.2558     |
|     Hard     | 0.0902    | 0.0012    | 1.0366     | 0.6266     | 0.7922     | 1.2231     | 0.6852     |
|   Average    | 0.1638    | 0.0292    | 1.0051     | 0.4294     | 0.6963     | 1.1418     | 0.3954     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2569    | 0.1287    | 0.9425     | 0.3009     | 0.5768     | 1.1180     | 0.2540     |
|   Moderate   | 0.2390    | 0.1128    | 0.9498     | 0.3025     | 0.6452     | 1.1187     | 0.2770     |
|     Hard     | 0.1980    | 0.0707    | 0.9953     | 0.3358     | 0.7084     | 1.2691     | 0.3336     |
|   Average    | 0.2313    | 0.1041    | 0.9625     | 0.3131     | 0.6435     | 1.1686     | 0.2882     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2437    | 0.1153    | 0.9422     | 0.3065     | 0.6281     | 1.1646     | 0.2624     |
|   Moderate   | 0.1839    | 0.0540    | 0.9829     | 0.3531     | 0.7916     | 1.4078     | 0.3035     |
|     Hard     | 0.1473    | 0.0335    | 0.9972     | 0.4335     | 0.8379     | 1.3463     | 0.4262     |
|   Average    | 0.1916    | 0.0676    | 0.9741     | 0.3644     | 0.7525     | 1.3062     | 0.3307     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2601    | 0.1318    | 0.9428     | 0.3026     | 0.5655     | 1.1073     | 0.2476     |
|   Moderate   | 0.2508    | 0.1257    | 0.9509     | 0.3040     | 0.6145     | 1.1483     | 0.2512     |
|     Hard     | 0.2452    | 0.1175    | 0.9515     | 0.3035     | 0.6337     | 1.1399     | 0.2471     |
|   Average    | 0.2520    | 0.1250    | 0.9484     | 0.3034     | 0.6046     | 1.1318     | 0.2486     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2262    | 0.0923    | 0.9462     | 0.3207     | 0.6667     | 1.2262     | 0.2659     |
|   Moderate   | 0.1862    | 0.0646    | 0.9428     | 0.4030     | 0.7247     | 1.3053     | 0.3901     |
|     Hard     | 0.1479    | 0.0303    | 0.9353     | 0.4714     | 0.7641     | 1.3877     | 0.5016     |
|   Average    | 0.1868    | 0.0624    | 0.9414     | 0.3984     | 0.7185     | 1.3064     | 0.3859     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3711    | 0.2650    | 0.8033     | 0.2837     | 0.4920     | 0.8171     | 0.2176     |
|   Moderate   | 0.3604    | 0.2511    | 0.8082     | 0.2857     | 0.5050     | 0.8275     | 0.2246     |
|     Hard     | 0.3433    | 0.2298    | 0.8279     | 0.2893     | 0.5197     | 0.8458     | 0.2332     |
|   Average    | 0.3583    | 0.2486    | 0.8131     | 0.2862     | 0.5056     | 0.8301     | 0.2251     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2500    | 0.1246    | 0.9432     | 0.3028     | 0.6222     | 1.1538     | 0.2543     |
|   Moderate   | 0.2457    | 0.1202    | 0.9496     | 0.3044     | 0.6282     | 1.1740     | 0.2618     |
|     Hard     | 0.2369    | 0.1097    | 0.9565     | 0.3093     | 0.6525     | 1.2140     | 0.2614     |
|   Average    | 0.2442    | 0.1181    | 0.9498     | 0.3055     | 0.6343     | 1.1806     | 0.2592     |



## References

```bib
@article{li2022bevformer,
  title = {BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author = {Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal = {arXiv preprint arXiv:2203.17270},
  year = {2022},
}
```

