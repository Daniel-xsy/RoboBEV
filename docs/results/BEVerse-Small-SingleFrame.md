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


## BEVerse-Small-SingleFrame

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.2682 | 0.1513  | 0.6631 | 0.4228 | 0.5406 | 1.3996 | 0.4483 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.1305    | 0.0340    | 0.8028     | 0.6164     | 0.7475     | 1.2273     | 0.6978     |
|   Frame Lost   | 0.0822    | 0.0274    | 0.8755     | 0.7651     | 0.8674     | 1.1223     | 0.8107     |
|  Color Quant   | 0.1002    | 0.0495    | 0.8923     | 0.7228     | 0.8517     | 1.1570     | 0.7850     |
|  Motion Blur   | 0.0716    | 0.0370    | 0.9117     | 0.7927     | 0.8818     | 1.1616     | 0.8833     |
|   Brightness   | 0.1336    | 0.0724    | 0.8340     | 0.6499     | 0.8086     | 1.2874     | 0.7333     |
|   Low Light    | 0.0132    | 0.0041    | 0.9862     | 0.9356     | 1.0175     | 0.9964     | 0.9707     |
|      Fog       | 0.0910    | 0.0406    | 0.8894     | 0.7200     | 0.8700     | 1.0564     | 0.8140     |
|      Snow      | 0.0116    | 0.0066    | 0.9785     | 0.9385     | 1.0000     | 1.0000     | 1.0000     |


## Experiment Log

> Time: Fri Jan 27 18:03:02 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1889    | 0.0569    | 0.7291     | 0.4955     | 0.6509     | 1.2742     | 0.5201     |
|   Moderate   | 0.1112    | 0.0175    | 0.8335     | 0.6378     | 0.7467     | 1.3031     | 0.7570     |
|     Hard     | 0.0915    | 0.0276    | 0.8457     | 0.7159     | 0.8450     | 1.1045     | 0.8163     |
|   Average    | 0.1305    | 0.0340    | 0.8028     | 0.6164     | 0.7475     | 1.2273     | 0.6978     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1962    | 0.0698    | 0.7196     | 0.4957     | 0.6518     | 1.3788     | 0.5197     |
|   Moderate   | 0.0504    | 0.0123    | 0.9070     | 0.7995     | 0.9504     | 0.9882     | 0.9123     |
|     Hard     | 0.0001    | 0.0002    | 1.0000     | 1.0000     | 1.0000     | 1.0000     | 1.0000     |
|   Average    | 0.0822    | 0.0274    | 0.8755     | 0.7651     | 0.8674     | 1.1223     | 0.8107     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2100    | 0.1117    | 0.7461     | 0.5007     | 0.6772     | 1.3872     | 0.5339     |
|   Moderate   | 0.0837    | 0.0355    | 0.9130     | 0.7279     | 0.8779     | 1.0838     | 0.8211     |
|     Hard     | 0.0068    | 0.0015    | 1.0179     | 0.9398     | 1.0000     | 1.0000     | 1.0000     |
|   Average    | 0.1002    | 0.0495    | 0.8923     | 0.7228     | 0.8517     | 1.1570     | 0.7850     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1727    | 0.0921    | 0.7926     | 0.5722     | 0.7189     | 1.4848     | 0.6500     |
|   Moderate   | 0.0317    | 0.0150    | 0.9636     | 0.8684     | 0.9266     | 1.0000     | 1.0000     |
|     Hard     | 0.0104    | 0.0040    | 0.9788     | 0.9376     | 1.0000     | 1.0000     | 1.0000     |
|   Average    | 0.0716    | 0.0370    | 0.9117     | 0.7927     | 0.8818     | 1.1616     | 0.8833     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2129    | 0.1145    | 0.7360     | 0.5022     | 0.6704     | 1.5247     | 0.5351     |
|   Moderate   | 0.1025    | 0.0643    | 0.8700     | 0.7234     | 0.8729     | 1.1718     | 0.8306     |
|     Hard     | 0.0856    | 0.0386    | 0.8959     | 0.7241     | 0.8825     | 1.1658     | 0.8342     |
|   Average    | 0.1336    | 0.0724    | 0.8340     | 0.6499     | 0.8086     | 1.2874     | 0.7333     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0304    | 0.0093    | 0.9705     | 0.8700     | 1.0526     | 0.9892     | 0.9122     |
|   Moderate   | 0.0091    | 0.0031    | 0.9880     | 0.9369     | 1.0000     | 1.0000     | 1.0000     |
|     Hard     | 0.0000    | 0.0000    | 1.0000     | 1.0000     | 1.0000     | 1.0000     | 1.0000     |
|   Average    | 0.0233    | 0.0073    | 0.9770     | 0.8922     | 1.0342     | 0.9924     | 0.9416     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1013    | 0.0569    | 0.8731     | 0.7200     | 0.8629     | 1.0581     | 0.8156     |
|   Moderate   | 0.0899    | 0.0385    | 0.8895     | 0.7201     | 0.8703     | 1.0551     | 0.8135     |
|     Hard     | 0.0817    | 0.0265    | 0.9056     | 0.7200     | 0.8769     | 1.0560     | 0.8129     |
|   Average    | 0.0910    | 0.0406    | 0.8894     | 0.7200     | 0.8700     | 1.0564     | 0.8140     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0154    | 0.0133    | 0.9752     | 0.9373     | 1.0000     | 1.0000     | 1.0000     |
|   Moderate   | 0.0097    | 0.0031    | 0.9792     | 0.9392     | 1.0000     | 1.0000     | 1.0000     |
|     Hard     | 0.0097    | 0.0033    | 0.9810     | 0.9390     | 1.0000     | 1.0000     | 1.0000     |
|   Average    | 0.0116    | 0.0066    | 0.9785     | 0.9385     | 1.0000     | 1.0000     | 1.0000     |



## References

```bib
@article{zhang2022beverse,
  title={Beverse: Unified perception and prediction in birds-eye-view for vision-centric autonomous driving},
  author={Zhang, Yunpeng and Zhu, Zheng and Zheng, Wenzhao and Huang, Junjie and Huang, Guan and Zhou, Jie and Lu, Jiwen},
  journal={arXiv preprint arXiv:2205.09743},
  year={2022}
}
```