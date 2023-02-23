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


## Sparse4D R101

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.5438  | 0.4409 | 0.6282 | 0.2721 | 0.3853 | 0.2922 | 0.1888 | 
| |
| Cam Crash      | 0.2873    | 0.1319    | 0.7852     | 0.2917     | 0.4989     | 0.9611     | 0.2510     |
| Frame Lost     | 0.2611    | 0.1050    | 0.8175     | 0.3166     | 0.5404     | 1.0253     | 0.2726     |
| Color Quant    | 0.3310    | 0.2345    | 0.8348     | 0.2956     | 0.5452     | 0.9712     | 0.2496     |
| Motion Blur    | 0.2514    | 0.1438    | 0.8719     | 0.3553     | 0.6780     | 1.0817     | 0.3347     |
| Brightness     | 0.3984    | 0.3296    | 0.7543     | 0.2835     | 0.4844     | 0.9232     | 0.2187     |
| Low Light      | 0.2510    | 0.1386    | 0.8501     | 0.3543     | 0.6464     | 1.1621     | 0.3356     |
| Fog            | 0.3884    | 0.3097    | 0.7552     | 0.2840     | 0.4933     | 0.9087     | 0.2229     |
| Snow           | 0.2259    | 0.1275    | 0.8860     | 0.3875     | 0.7116     | 1.1418     | 0.3936     |


## Experiment Log

> Time: Day Month xx xx:xx:xx 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3369    | 0.2035    | 0.7533     | 0.2857     | 0.4744     | 0.9023     | 0.2325     |
| Moderate     | 0.2623    | 0.0979    | 0.8150     | 0.2933     | 0.5016     | 1.0038     | 0.2571     |
| Hard         | 0.2628    | 0.0944    | 0.7874     | 0.2962     | 0.5206     | 0.9772     | 0.2633     |
| Average      | 0.2873    | 0.1319    | 0.7852     | 0.2917     | 0.4989     | 0.9611     | 0.2510     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3494    | 0.2259    | 0.7498     | 0.2847     | 0.4649     | 0.9060     | 0.2306     |
| Moderate     | 0.2479    | 0.0746    | 0.8204     | 0.3008     | 0.5351     | 0.9929     | 0.2449     |
| Hard         | 0.1861    | 0.0143    | 0.8824     | 0.3643     | 0.6212     | 1.1770     | 0.3423     |
| Average      | 0.2611    | 0.1050    | 0.8175     | 0.3166     | 0.5404     | 1.0253     | 0.2726     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4109    | 0.3427    | 0.7488     | 0.2801     | 0.4620     | 0.8837     | 0.2295     |
| Moderate     | 0.3385    | 0.2462    | 0.8222     | 0.2899     | 0.5613     | 0.9295     | 0.2433     |
| Hard         | 0.2435    | 0.1147    | 0.9333     | 0.3167     | 0.6124     | 1.1004     | 0.2761     |
| Average      | 0.3310    | 0.2345    | 0.8348     | 0.2956     | 0.5452     | 0.9712     | 0.2496     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3692    | 0.2830    | 0.7731     | 0.2898     | 0.5352     | 0.8952     | 0.2292     |
| Moderate     | 0.2169    | 0.0927    | 0.9302     | 0.3286     | 0.7464     | 1.0806     | 0.2895     |
| Hard         | 0.1681    | 0.0557    | 0.9124     | 0.4475     | 0.7525     | 1.2693     | 0.4853     |
| Average      | 0.2514    | 0.1438    | 0.8719     | 0.3553     | 0.6780     | 1.0817     | 0.3347     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4273    | 0.3626    | 0.7157     | 0.2801     | 0.4501     | 0.8722     | 0.2219     |
| Moderate     | 0.3991    | 0.3321    | 0.7538     | 0.2838     | 0.4918     | 0.9238     | 0.2157     |
| Hard         | 0.3687    | 0.2942    | 0.7933     | 0.2866     | 0.5114     | 0.9737     | 0.2186     |
| Average      | 0.3984    | 0.3296    | 0.7543     | 0.2835     | 0.4844     | 0.9232     | 0.2187     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3115    | 0.2096    | 0.8259     | 0.3005     | 0.5749     | 0.9894     | 0.2419     |
| Moderate     | 0.2613    | 0.1398    | 0.8561     | 0.3154     | 0.6396     | 1.1260     | 0.2751     |
| Hard         | 0.1803    | 0.0664    | 0.8682     | 0.4471     | 0.7246     | 1.3708     | 0.4897     |
| Average      | 0.2510    | 0.1386    | 0.8501     | 0.3543     | 0.6464     | 1.1621     | 0.3356     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4021    | 0.3294    | 0.7434     | 0.2836     | 0.4893     | 0.8918     | 0.2179     |
| Moderate     | 0.3926    | 0.3128    | 0.7516     | 0.2828     | 0.4819     | 0.8990     | 0.2228     |
| Hard         | 0.3706    | 0.2869    | 0.7706     | 0.2857     | 0.5087     | 0.9353     | 0.2280     |
| Average      | 0.3884    | 0.3097    | 0.7552     | 0.2840     | 0.4933     | 0.9087     | 0.2229     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2259    | 0.1275    | 0.8860     | 0.3875     | 0.7116     | 1.1418     | 0.3936     |
| Moderate     | 0.1757    | 0.0736    | 0.9395     | 0.4159     | 0.8436     | 1.2643     | 0.4121     |
| Hard         | 0.1682    | 0.0654    | 0.9548     | 0.4160     | 0.8515     | 1.2578     | 0.4229     |
| Average      | 0.1899    | 0.0888    | 0.9268     | 0.4065     | 0.8022     | 1.2213     | 0.4095     |



## References

```bib
@article{lin2022sparse4d,
  title={Sparse4D: Multi-view 3D Object Detection with Sparse Spatial-Temporal Fusion},
  author={Lin, Xuewu and Lin, Tianwei and Pei, Zixiang and Huang, Lichao and Su, Zhizhong},
  journal={arXiv preprint arXiv:2211.10581},
  year={2022}
}
```
