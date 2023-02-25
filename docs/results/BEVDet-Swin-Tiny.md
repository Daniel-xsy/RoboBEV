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


## BEVDet-swin-tiny

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4037 | 0.3080 | 0.6648 | 0.2729 | 0.5323 | 0.8278 | 0.2050 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.2609    | 0.1053    | 0.7786     | 0.3246     | 0.5761     | 0.9821     | 0.2822     |
|   Frame Lost   | 0.2115    | 0.0826    | 0.8174     | 0.4207     | 0.6710     | 1.0138     | 0.4294     |
|  Color Quant   | 0.2278    | 0.1487    | 0.8236     | 0.4518     | 0.7461     | 1.1668     | 0.4742     |
|  Motion Blur   | 0.2128    | 0.1235    | 0.8455     | 0.4457     | 0.7074     | 1.1857     | 0.5080     |
|   Brightness   | 0.2191    | 0.1370    | 0.8300     | 0.4523     | 0.7277     | 1.2995     | 0.4833     |
|   Low Light    | 0.0490    | 0.0180    | 0.9883     | 0.7696     | 1.0083     | 1.1225     | 0.8607     |
|      Fog       | 0.2450    | 0.1396    | 0.8459     | 0.3656     | 0.6839     | 1.2694     | 0.3520     |
|      Snow      | 0.0680    | 0.0312    | 0.9730     | 0.7665     | 0.8973     | 1.2609     | 0.8393     |


## Experiment Log

> Time: Fri Feb 24 19:37:37 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3125    | 0.1667    | 0.7428     | 0.2849     | 0.5806     | 0.8800     | 0.2201     |
|   Moderate   | 0.2328    | 0.0731    | 0.8256     | 0.3425     | 0.5669     | 1.0773     | 0.3019     |
|     Hard     | 0.2373    | 0.0762    | 0.7675     | 0.3463     | 0.5809     | 0.9891     | 0.3246     |
|   Average    | 0.2609    | 0.1053    | 0.7786     | 0.3246     | 0.5761     | 0.9821     | 0.2822     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3245    | 0.1821    | 0.7239     | 0.2775     | 0.5814     | 0.8791     | 0.2037     |
|   Moderate   | 0.2180    | 0.0556    | 0.8139     | 0.3420     | 0.6295     | 1.0409     | 0.3125     |
|     Hard     | 0.0920    | 0.0102    | 0.9145     | 0.6425     | 0.8021     | 1.1214     | 0.7719     |
|   Average    | 0.2115    | 0.0826    | 0.8174     | 0.4207     | 0.6710     | 1.0138     | 0.4294     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3663    | 0.2685    | 0.6904     | 0.2770     | 0.5618     | 0.9104     | 0.2393     |
|   Moderate   | 0.2520    | 0.1513    | 0.7961     | 0.3625     | 0.6922     | 1.1799     | 0.3854     |
|     Hard     | 0.0649    | 0.0263    | 0.9843     | 0.7159     | 0.9842     | 1.4101     | 0.7979     |
|   Average    | 0.2278    | 0.1487    | 0.8236     | 0.4518     | 0.7461     | 1.1668     | 0.4742     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3418    | 0.2385    | 0.7391     | 0.2849     | 0.5824     | 0.9489     | 0.2189     |
|   Moderate   | 0.1722    | 0.0846    | 0.8716     | 0.4882     | 0.7527     | 1.3531     | 0.5882     |
|     Hard     | 0.1244    | 0.0475    | 0.9257     | 0.5641     | 0.7872     | 1.2551     | 0.7168     |
|   Average    | 0.2128    | 0.1235    | 0.8455     | 0.4457     | 0.7074     | 1.1857     | 0.5080     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3292    | 0.2258    | 0.7301     | 0.2936     | 0.5961     | 1.0621     | 0.2176     |
|   Moderate   | 0.2127    | 0.1183    | 0.8289     | 0.4238     | 0.7548     | 1.5022     | 0.4571     |
|     Hard     | 0.1156    | 0.0668    | 0.9310     | 0.6396     | 0.8323     | 1.3342     | 0.7753     |
|   Average    | 0.2191    | 0.1370    | 0.8300     | 0.4523     | 0.7277     | 1.2995     | 0.4833     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.0649    | 0.0303    | 0.9777     | 0.7249     | 0.9451     | 1.1186     | 0.8546     |
|   Moderate   | 0.0447    | 0.0182    | 0.9918     | 0.7915     | 1.0292     | 1.1262     | 0.8607     |
|     Hard     | 0.0373    | 0.0056    | 0.9955     | 0.7925     | 1.0506     | 1.1226     | 0.8669     |
|   Average    | 0.0490    | 0.0180    | 0.9883     | 0.7696     | 1.0083     | 1.1225     | 0.8607     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2769    | 0.1742    | 0.8119     | 0.3349     | 0.6491     | 1.1858     | 0.3061     |
|   Moderate   | 0.2504    | 0.1370    | 0.8493     | 0.3425     | 0.6720     | 1.2905     | 0.3168     |
|     Hard     | 0.2078    | 0.1075    | 0.8765     | 0.4194     | 0.7307     | 1.3318     | 0.4331     |
|   Average    | 0.2450    | 0.1396    | 0.8459     | 0.3656     | 0.6839     | 1.2694     | 0.3520     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1342    | 0.0578    | 0.9422     | 0.5731     | 0.7594     | 1.4467     | 0.6721     |
|   Moderate   | 0.0337    | 0.0162    | 0.9872     | 0.8634     | 0.9679     | 1.1824     | 0.9249     |
|     Hard     | 0.0360    | 0.0197    | 0.9897     | 0.8630     | 0.9645     | 1.1535     | 0.9209     |
|   Average    | 0.0680    | 0.0312    | 0.9730     | 0.7665     | 0.8973     | 1.2609     | 0.8393     |



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