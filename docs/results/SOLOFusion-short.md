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


## SOLOFusion-Short-Only

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.3907    | 0.3438    | 0.6691     | 0.2809     | 0.6638     | 0.8803     | 0.3180     |
| |
| Cam Crash      | 0.2541    | 0.1132    | 0.7542     | 0.2848     | 0.7337     | 0.9248     | 0.3273     |
| Frame Lost     | 0.2195    | 0.0848    | 0.8066     | 0.3285     | 0.7407     | 1.0092     | 0.3785     |
| Color Quant    | 0.2804    | 0.2013    | 0.7790     | 0.3214     | 0.7702     | 0.9825     | 0.3706     |
| Motion Blur    | 0.2603    | 0.1717    | 0.8145     | 0.2968     | 0.8353     | 0.9831     | 0.3414     |
| Brightness     | 0.2966    | 0.2339    | 0.7497     | 0.3258     | 0.8038     | 1.0663     | 0.3433     |
| Low Light      | 0.2033    | 0.1138    | 0.7744     | 0.3716     | 0.9146     | 1.1518     | 0.4757     |
| Fog            | 0.2998    | 0.2260    | 0.7556     | 0.2908     | 0.7761     | 1.0074     | 0.3238     |
| Snow           | 0.1066    | 0.0427    | 0.9399     | 0.5888     | 0.9026     | 1.1212     | 0.7160     |


## Experiment Log

> Time: Thu Apr  6 15:58:22 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2946    | 0.1805    | 0.7143     | 0.2844     | 0.7247     | 0.9125     | 0.3202     |
| Moderate     | 0.2296    | 0.0794    | 0.7853     | 0.2846     | 0.7400     | 0.9578     | 0.3334     |
| Hard         | 0.2382    | 0.0797    | 0.7630     | 0.2853     | 0.7363     | 0.9042     | 0.3282     |
| Average      | 0.2541    | 0.1132    | 0.7542     | 0.2848     | 0.7337     | 0.9248     | 0.3273     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3019    | 0.1932    | 0.7177     | 0.2832     | 0.7040     | 0.9245     | 0.3178     |
| Moderate     | 0.2082    | 0.0535    | 0.8102     | 0.2889     | 0.7615     | 1.0239     | 0.3254     |
| Hard         | 0.1484    | 0.0076    | 0.8918     | 0.4135     | 0.7566     | 1.0792     | 0.4924     |
| Average      | 0.2195    | 0.0848    | 0.8066     | 0.3285     | 0.7407     | 1.0092     | 0.3785     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3686    | 0.3081    | 0.6873     | 0.2832     | 0.6657     | 0.9008     | 0.3174     |
| Moderate     | 0.2945    | 0.2136    | 0.7636     | 0.2928     | 0.7506     | 0.9831     | 0.3327     |
| Hard         | 0.1781    | 0.0822    | 0.8860     | 0.3883     | 0.8942     | 1.0636     | 0.4618     |
| Average      | 0.2804    | 0.2013    | 0.7790     | 0.3214     | 0.7702     | 0.9825     | 0.3706     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3518    | 0.2891    | 0.7071     | 0.2851     | 0.7170     | 0.9035     | 0.3147     |
| Moderate     | 0.2321    | 0.1369    | 0.8425     | 0.2990     | 0.8748     | 1.0024     | 0.3471     |
| Hard         | 0.1969    | 0.0892    | 0.8939     | 0.3064     | 0.9141     | 1.0433     | 0.3624     |
| Average      | 0.2603    | 0.1717    | 0.8145     | 0.2968     | 0.8353     | 0.9831     | 0.3414     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3537    | 0.2970    | 0.7028     | 0.2908     | 0.7037     | 0.9436     | 0.3070     |
| Moderate     | 0.2955    | 0.2234    | 0.7582     | 0.3054     | 0.7921     | 1.1070     | 0.3068     |
| Hard         | 0.2406    | 0.1814    | 0.7881     | 0.3812     | 0.9157     | 1.1483     | 0.4162     |
| Average      | 0.2966    | 0.2339    | 0.7497     | 0.3258     | 0.8038     | 1.0663     | 0.3433     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2554    | 0.1644    | 0.7548     | 0.3055     | 0.8403     | 1.0493     | 0.3674     |
| Moderate     | 0.2030    | 0.1197    | 0.7744     | 0.3802     | 0.9359     | 1.1085     | 0.4777     |
| Hard         | 0.1514    | 0.0573    | 0.7939     | 0.4291     | 0.9676     | 1.2976     | 0.5819     |
| Average      | 0.2033    | 0.1138    | 0.7744     | 0.3716     | 0.9146     | 1.1518     | 0.4757     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3236    | 0.2555    | 0.7314     | 0.2873     | 0.7604     | 0.9569     | 0.3051     |
| Moderate     | 0.2992    | 0.2278    | 0.7558     | 0.2908     | 0.7749     | 1.0120     | 0.3258     |
| Hard         | 0.2767    | 0.1948    | 0.7796     | 0.2942     | 0.7931     | 1.0534     | 0.3406     |
| Average      | 0.2998    | 0.2260    | 0.7556     | 0.2908     | 0.7761     | 1.0074     | 0.3238     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.1782    | 0.0836    | 0.8725     | 0.3855     | 0.8876     | 1.0505     | 0.4901     |
| Moderate     | 0.0702    | 0.0236    | 0.9631     | 0.7231     | 0.8706     | 1.1304     | 0.8590     |
| Hard         | 0.0714    | 0.0210    | 0.9842     | 0.6578     | 0.9497     | 1.1827     | 0.7990     |
| Average      | 0.1066    | 0.0427    | 0.9399     | 0.5888     | 0.9026     | 1.1212     | 0.7160     |



## References

```bib
@article{Park2022TimeWT,
  title={Time Will Tell: New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection},
  author={Park, Jinhyung and Xu, Chenfeng and Yang, Shijia and Keutzer, Kurt and Kitani, Kris and Tomizuka, Masayoshi and Zhan, Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
