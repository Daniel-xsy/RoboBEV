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


## PolarFormer-R101

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.4602 | 0.3916 | 0.7060 | 0.2718 | 0.3610 | 0.8079 | 0.2093 |
| |
| Cam Crash      | 0.3133    | 0.1425    | 0.7746     | 0.2840     | 0.4440     | 0.8524     | 0.2250     |
| Frame Lost     | 0.2808    | 0.1134    | 0.8034     | 0.3093     | 0.4981     | 0.8988     | 0.2498     |
| Color Quant    | 0.3509    | 0.2538    | 0.8059     | 0.2999     | 0.4812     | 0.9724     | 0.2592     |
| Motion Blur    | 0.3221    | 0.2117    | 0.8196     | 0.2946     | 0.5727     | 0.9379     | 0.2258     |
| Brightness     | 0.4304    | 0.3574    | 0.7390     | 0.2738     | 0.4149     | 0.8522     | 0.2032     |
| Low Light      |  |  |  |  |  |  |  |
| Fog            | 0.4262    | 0.3518    | 0.7338     | 0.2735     | 0.4143     | 0.8672     | 0.2082     |
| Snow           | 0.2304    | 0.1058    | 0.9125     | 0.3363     | 0.6592     | 1.2284     | 0.3174     |


## Experiment Log

> Time: Fri Feb 17 18:56:16 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3619    | 0.2181    | 0.7442     | 0.2770     | 0.3972     | 0.8414     | 0.2112     |
| Moderate     | 0.2849    | 0.1004    | 0.8018     | 0.2877     | 0.4279     | 0.9142     | 0.2217     |
| Hard         | 0.2930    | 0.1092    | 0.7777     | 0.2874     | 0.5069     | 0.8017     | 0.2420     |
| Average      | 0.3133    | 0.1425    | 0.7746     | 0.2840     | 0.4440     | 0.8524     | 0.2250     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3761    | 0.2419    | 0.7369     | 0.2759     | 0.3936     | 0.8353     | 0.2071     |
| Moderate     | 0.2734    | 0.0817    | 0.7985     | 0.2900     | 0.4869     | 0.8785     | 0.2212     |
| Hard         | 0.1929    | 0.0166    | 0.8747     | 0.3619     | 0.6138     | 0.9827     | 0.3210     |
| Average      | 0.2808    | 0.1134    | 0.8034     | 0.3093     | 0.4981     | 0.8988     | 0.2498     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4364    | 0.3618    | 0.7339     | 0.2726     | 0.3862     | 0.8410     | 0.2111     |
| Moderate     | 0.3681    | 0.2703    | 0.7991     | 0.2785     | 0.4676     | 0.8986     | 0.2265     |
| Hard         | 0.2483    | 0.1293    | 0.8848     | 0.3487     | 0.5898     | 1.1776     | 0.3401     |
| Average      | 0.3509    | 0.2538    | 0.8059     | 0.2999     | 0.4812     | 0.9724     | 0.2592     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4197    | 0.3333    | 0.7282     | 0.2766     | 0.4191     | 0.8390     | 0.2064     |
| Moderate     | 0.3006    | 0.1807    | 0.8348     | 0.2957     | 0.6035     | 0.9351     | 0.2281     |
| Hard         | 0.2459    | 0.1210    | 0.8957     | 0.3116     | 0.6955     | 1.0397     | 0.2430     |
| Average      | 0.3221    | 0.2117    | 0.8196     | 0.2946     | 0.5727     | 0.9379     | 0.2258     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4534    | 0.3840    | 0.7119     | 0.2717     | 0.3816     | 0.8180     | 0.2023     |
| Moderate     | 0.4313    | 0.3587    | 0.7418     | 0.2736     | 0.4046     | 0.8606     | 0.1998     |
| Hard         | 0.4065    | 0.3296    | 0.7632     | 0.2762     | 0.4584     | 0.8780     | 0.2074     |
| Average      | 0.4304    | 0.3574    | 0.7390     | 0.2738     | 0.4149     | 0.8522     | 0.2032     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         |  |  |  |  |  |  |  |
| Moderate     |  |  |  |  |  |  |  |
| Hard         |  |  |  |  |  |  |  |
| Average      |  |  |  |  |  |  |  |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4414    | 0.3684    | 0.7242     | 0.2719     | 0.3911     | 0.8395     | 0.2016     |
| Moderate     | 0.4291    | 0.3552    | 0.7261     | 0.2730     | 0.4094     | 0.8677     | 0.2091     |
| Hard         | 0.4082    | 0.3319    | 0.7512     | 0.2755     | 0.4424     | 0.8943     | 0.2140     |
| Average      | 0.4262    | 0.3518    | 0.7338     | 0.2735     | 0.4143     | 0.8672     | 0.2082     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2828    | 0.1619    | 0.8581     | 0.3101     | 0.5360     | 1.1919     | 0.2776     |
| Moderate     | 0.2014    | 0.0729    | 0.9367     | 0.3594     | 0.7171     | 1.2392     | 0.3374     |
| Hard         | 0.2069    | 0.0827    | 0.9426     | 0.3394     | 0.7246     | 1.2540     | 0.3373     |
| Average      | 0.2304    | 0.1058    | 0.9125     | 0.3363     | 0.6592     | 1.2284     | 0.3174     |



## References

```bib
@article{jiang2022polarformer,
  title={Polarformer: Multi-camera 3d object detection with polar transformers},
  author={Jiang, Yanqin and Zhang, Li and Miao, Zhenwei and Zhu, Xiatian and Gao, Jin and Hu, Weiming and Jiang, Yu-Gang},
  journal={arXiv preprint arXiv:2206.15398},
  year={2022}
}
```
