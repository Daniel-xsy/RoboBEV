<img src="../figs/logo2.png" align="right" width="30%">

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


## BEVFormer-Base

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean       | 0.5174    | 0.4164    | 0.6726     | 0.2734     | 0.3704     | 0.3941     | 0.1974     |
| |
| Cam Crash   | 0.3154    | 0.1545    | 0.8015     | 0.2975     | 0.5031     | 0.7865     | 0.2301     |
| Frame Lost  | 0.3017    | 0.1307    | 0.8359     | 0.3053     | 0.5262     | 0.7364     | 0.2328     |
| Color Quant | 0.3509    | 0.2393    | 0.8294     | 0.2953     | 0.5200     | 0.8079     | 0.2350     |
| Motion Blur | 0.2695    | 0.1531    | 0.8739     | 0.3236     | 0.6941     | 0.9334     | 0.2592     |
| Brightness  | 0.4184    | 0.3312    | 0.7457     | 0.2832     | 0.4721     | 0.7686     | 0.2024     |
| Low Light   | 0.2961    | 0.1866    | 0.8343     | 0.3101     | 0.6297     | 0.9348     | 0.2644     |
| Fog         | 0.4069    | 0.3141    | 0.7627     | 0.2837     | 0.4711     | 0.7798     | 0.2046     |
| Snow        | 0.1857    | 0.0739    | 0.9405     | 0.3966     | 0.7806     | 1.0880     | 0.3951     |


## Experiment Log

> Time: Sun Feb 12 21:42:23 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3719    | 0.2324    | 0.7753     | 0.2883     | 0.4503     | 0.7161     | 0.2127     |
| Moderate     | 0.3100    | 0.1370    | 0.8431     | 0.2965     | 0.4819     | 0.7323     | 0.2315     |
| Hard         | 0.2643    | 0.0941    | 0.7862     | 0.3077     | 0.5772     | 0.9110     | 0.2460     |
| Average      | 0.3154    | 0.1545    | 0.8015     | 0.2975     | 0.5031     | 0.7865     | 0.2301     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3818    | 0.2552    | 0.7683     | 0.2843     | 0.4490     | 0.7455     | 0.2110     |
| Moderate     | 0.2878    | 0.1037    | 0.8411     | 0.3056     | 0.4982     | 0.7616     | 0.2345     |
| Hard         | 0.2355    | 0.0331    | 0.8983     | 0.3259     | 0.6314     | 0.7021     | 0.2528     |
| Average      | 0.3017    | 0.1307    | 0.8359     | 0.3053     | 0.5262     | 0.7364     | 0.2328     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4323    | 0.3447    | 0.7454     | 0.2780     | 0.4292     | 0.7385     | 0.2088     |
| Moderate     | 0.3651    | 0.2544    | 0.8038     | 0.2862     | 0.5050     | 0.7943     | 0.2318     |
| Hard         | 0.2553    | 0.1189    | 0.9389     | 0.3217     | 0.6257     | 0.8908     | 0.2643     |
| Average      | 0.3509    | 0.2393    | 0.8294     | 0.2953     | 0.5200     | 0.8079     | 0.2350     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3918    | 0.2968    | 0.7602     | 0.2859     | 0.5047     | 0.7998     | 0.2152     |
| Moderate     | 0.2311    | 0.1022    | 0.8999     | 0.3276     | 0.7440     | 0.9607     | 0.2679     |
| Hard         | 0.1855    | 0.0603    | 0.9615     | 0.3574     | 0.8336     | 1.0397     | 0.2944     |
| Average      | 0.2695    | 0.1531    | 0.8739     | 0.3236     | 0.6941     | 0.9334     | 0.2592     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4466    | 0.3656    | 0.7313     | 0.2793     | 0.4288     | 0.7224     | 0.2001     |
| Moderate     | 0.4201    | 0.3316    | 0.7444     | 0.2840     | 0.4664     | 0.7612     | 0.2009     |
| Hard         | 0.3885    | 0.2965    | 0.7614     | 0.2862     | 0.5212     | 0.8222     | 0.2061     |
| Average      | 0.4184    | 0.3312    | 0.7457     | 0.2832     | 0.4721     | 0.7686     | 0.2024     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3161    | 0.2098    | 0.8198     | 0.3035     | 0.6155     | 0.9016     | 0.2482     |
| Moderate     | 0.3178    | 0.2098    | 0.8138     | 0.3028     | 0.6042     | 0.9008     | 0.2498     |
| Hard         | 0.2543    | 0.1402    | 0.8692     | 0.3239     | 0.6695     | 1.0020     | 0.2952     |
| Average      | 0.2961    | 0.1866    | 0.8343     | 0.3101     | 0.6297     | 0.9348     | 0.2644     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4207    | 0.3318    | 0.7545     | 0.2806     | 0.4608     | 0.7595     | 0.1968     |
| Moderate     | 0.4092    | 0.3175    | 0.7601     | 0.2827     | 0.4703     | 0.7770     | 0.2051     |
| Hard         | 0.3906    | 0.2930    | 0.7734     | 0.2879     | 0.4822     | 0.8029     | 0.2120     |
| Average      | 0.4069    | 0.3141    | 0.7627     | 0.2837     | 0.4711     | 0.7798     | 0.2046     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2296    | 0.1060    | 0.9212     | 0.3424     | 0.6615     | 1.0302     | 0.3087     |
| Moderate     | 0.1702    | 0.0621    | 0.9353     | 0.4245     | 0.8143     | 1.0902     | 0.4346     |
| Hard         | 0.1572    | 0.0536    | 0.9649     | 0.4229     | 0.8661     | 1.1435     | 0.4419     |
| Average      | 0.1857    | 0.0739    | 0.9405     | 0.3966     | 0.7806     | 1.0880     | 0.3951     |



## References

```bib
@article{li2022bevformer,
  title = {BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author = {Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal = {arXiv preprint arXiv:2203.17270},
  year = {2022},
}
```
