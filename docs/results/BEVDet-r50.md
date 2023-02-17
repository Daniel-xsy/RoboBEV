<img src="F:\Research\Robust BEV Detection\Robust-BEV-Detection\docs\figs\logo2.png" align="right" width="30%">

# RoboDet Benchmark

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


## Model Name

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.3770 | 0.2987 | 0.7336 | 0.2744 | 0.5713 | 0.9051 | 0.2394 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.2486    | 0.0990    | 0.8147     | 0.2975     | 0.6402     | 0.9990     | 0.2842     |
|   Frame Lost   | 0.1924    | 0.0781    | 0.8545     | 0.4413     | 0.7179     | 1.0247     | 0.4780     |
|  Color Quant   | 0.2408    | 0.1542    | 0.8718     | 0.3579     | 0.7376     | 1.2194     | 0.3958     |
|  Motion Blur   | 0.2061    | 0.1156    | 0.8891     | 0.4020     | 0.7693     | 1.1521     | 0.4645     |
|   Brightness   | 0.2565    | 0.1787    | 0.8380     | 0.3736     | 0.7216     | 1.2912     | 0.3955     |
|   Low Light    | 0.1493    | 0.0657    | 1.0022     | 0.4002     | 0.9416     | 1.2742     | 0.4976     |
|      Fog       | 0.2461    | 0.1404    | 0.8801     | 0.3018     | 0.7483     | 1.1610     | 0.3112     |
|      Snow      | 0.0625    | 0.0254    | 0.9853     | 0.7204     | 1.0029     | 1.1642     | 0.8160     |


## Experiment Log

> Time: Mon Feb 13 15:52:59 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2905    | 0.1581    | 0.7915     | 0.2749     | 0.6302     | 0.9313     | 0.2574     |
|   Moderate   | 0.2223    | 0.0679    | 0.8403     | 0.3309     | 0.6208     | 1.0798     | 0.3247     |
|     Hard     | 0.2329    | 0.0709    | 0.8124     | 0.2868     | 0.6697     | 0.9858     | 0.2706     |
|   Average    | 0.2486    | 0.0990    | 0.8147     | 0.2975     | 0.6402     | 0.9990     | 0.2842     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3027    | 0.1752    | 0.7857     | 0.2774     | 0.6143     | 0.9262     | 0.2451     |
|   Moderate   | 0.2076    | 0.0508    | 0.8516     | 0.3338     | 0.6586     | 1.1364     | 0.3342     |
|     Hard     | 0.0667    | 0.0083    | 0.9261     | 0.7126     | 0.8808     | 1.0116     | 0.8546     |
|   Average    | 0.1924    | 0.0781    | 0.8545     | 0.4413     | 0.7179     | 1.0247     | 0.4780     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3373    | 0.2608    | 0.7665     | 0.2770     | 0.6201     | 1.0288     | 0.2678     |
|   Moderate   | 0.2450    | 0.1599    | 0.8534     | 0.3639     | 0.7289     | 1.2417     | 0.4031     |
|     Hard     | 0.1400    | 0.0419    | 0.9956     | 0.4329     | 0.8639     | 1.3876     | 0.5166     |
|   Average    | 0.2408    | 0.1542    | 0.8718     | 0.3579     | 0.7376     | 1.2194     | 0.3958     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3172    | 0.2214    | 0.7986     | 0.2796     | 0.6347     | 0.9760     | 0.2457     |
|   Moderate   | 0.1816    | 0.0786    | 0.9320     | 0.3637     | 0.8502     | 1.1827     | 0.4312     |
|     Hard     | 0.1194    | 0.0467    | 0.9366     | 0.5627     | 0.8231     | 1.2976     | 0.7166     |
|   Average    | 0.2061    | 0.1156    | 0.8891     | 0.4020     | 0.7693     | 1.1521     | 0.4645     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3226    | 0.2420    | 0.7818     | 0.2834     | 0.6731     | 1.0987     | 0.2456     |
|   Moderate   | 0.2415    | 0.1713    | 0.8363     | 0.4140     | 0.7245     | 1.3378     | 0.4670     |
|     Hard     | 0.2053    | 0.1228    | 0.8959     | 0.4233     | 0.7673     | 1.4370     | 0.4739     |
|   Average    | 0.2565    | 0.1787    | 0.8380     | 0.3736     | 0.7216     | 1.2912     | 0.3955     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1587    | 0.0745    | 1.0107     | 0.3838     | 0.9373     | 1.1912     | 0.4643     |
|   Moderate   | 0.1587    | 0.0740    | 1.0063     | 0.3833     | 0.9349     | 1.1845     | 0.4653     |
|     Hard     | 0.1305    | 0.0487    | 0.9897     | 0.4336     | 0.9527     | 1.4470     | 0.5632     |
|   Average    | 0.1493    | 0.0657    | 1.0022     | 0.4002     | 0.9416     | 1.2742     | 0.4976     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2780    | 0.1790    | 0.8460     | 0.2798     | 0.7233     | 1.0716     | 0.2658     |
|   Moderate   | 0.2477    | 0.1364    | 0.8822     | 0.2888     | 0.7533     | 1.1280     | 0.2809     |
|     Hard     | 0.2125    | 0.1059    | 0.9122     | 0.3368     | 0.7684     | 1.2835     | 0.3868     |
|   Average    | 0.2461    | 0.1404    | 0.8801     | 0.3018     | 0.7483     | 1.1610     | 0.3112     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1206    | 0.0466    | 0.9653     | 0.5091     | 0.9417     | 1.2147     | 0.6114     |
|   Moderate   | 0.0420    | 0.0200    | 0.9896     | 0.7920     | 1.0515     | 1.1336     | 0.8985     |
|     Hard     | 0.0249    | 0.0095    | 1.0010     | 0.8601     | 1.0156     | 1.1442     | 0.9381     |
|   Average    | 0.0625    | 0.0254    | 0.9853     | 0.7204     | 1.0029     | 1.1642     | 0.8160     |



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