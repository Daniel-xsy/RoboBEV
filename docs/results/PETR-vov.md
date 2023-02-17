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


## PETR-VovNet

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean       | 0.4550    | 0.4035    | 0.7362     | 0.2710     | 0.4316     | 0.8249     | 0.2039     |
| Motion Blur | 0.2490    | 0.1395    | 0.9521     | 0.3153     | 0.7424     | 1.0353     | 0.2639     |
| Color Quant | 0.2968    | 0.2089    | 0.8818     | 0.3455     | 0.5997     | 1.0875     | 0.3123     |
| Frame Lost  | 0.2792    | 0.1153    | 0.8311     | 0.2909     | 0.5662     | 0.8816     | 0.2144     |
| Cam Crash   | 0.2924    | 0.1408    | 0.8167     | 0.2854     | 0.5492     | 0.9014     | 0.2267     |
| Brightness  | 0.3858    | 0.3199    | 0.7982     | 0.2779     | 0.5256     | 0.9342     | 0.2112     |
| Low Light   | 0.2791    | 0.1674    | 0.8616     | 0.3038     | 0.6158     | 1.1423     | 0.2652     |
| Fog         | 0.3703    | 0.2815    | 0.8337     | 0.2778     | 0.4982     | 0.8833     | 0.2111     |
| Snow        | 0.2632    | 0.1653    | 0.8980     | 0.3138     | 0.7034     | 1.1314     | 0.2886     |


## Experiment Log

> Time: Fri Jan 20 23:39:21 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3348    | 0.2107    | 0.8052     | 0.2819     | 0.5253     | 0.8717     | 0.2216     |
| Moderate     | 0.2630    | 0.0998    | 0.8451     | 0.2836     | 0.5406     | 0.9566     | 0.2431     |
| Hard         | 0.2795    | 0.1118    | 0.7998     | 0.2907     | 0.5817     | 0.8759     | 0.2155     |
| Average      | 0.2924    | 0.1408    | 0.8167     | 0.2854     | 0.5492     | 0.9014     | 0.2267     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3619    | 0.2459    | 0.7765     | 0.2761     | 0.4816     | 0.8678     | 0.2083     |
| Moderate     | 0.2618    | 0.0828    | 0.8323     | 0.2908     | 0.5614     | 0.8974     | 0.2143     |
| Hard         | 0.2140    | 0.0171    | 0.8846     | 0.3059     | 0.6556     | 0.8795     | 0.2205     |
| Average      | 0.2792    | 0.1153    | 0.8311     | 0.2909     | 0.5662     | 0.8816     | 0.2144     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4254    | 0.3647    | 0.7639     | 0.2748     | 0.4584     | 0.8558     | 0.2171     |
| Moderate     | 0.3156    | 0.2212    | 0.8591     | 0.2954     | 0.5736     | 0.9774     | 0.2447     |
| Hard         | 0.1495    | 0.0408    | 1.0224     | 0.4662     | 0.7670     | 1.4292     | 0.4752     |
| Average      | 0.2968    | 0.2089    | 0.8818     | 0.3455     | 0.5997     | 1.0875     | 0.3123     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3826    | 0.3023    | 0.8319     | 0.2758     | 0.5137     | 0.8587     | 0.2049     |
| Moderate     | 0.2067    | 0.0793    | 0.9682     | 0.3181     | 0.7665     | 1.0943     | 0.2767     |
| Hard         | 0.1575    | 0.0369    | 1.0563     | 0.3519     | 0.9471     | 1.1529     | 0.3102     |
| Average      | 0.2490    | 0.1395    | 0.9521     | 0.3153     | 0.7424     | 1.0353     | 0.2639     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4348    | 0.3750    | 0.7586     | 0.2718     | 0.4595     | 0.8336     | 0.2035     |
| Moderate     | 0.3785    | 0.3121    | 0.8002     | 0.2788     | 0.5339     | 0.9532     | 0.2099     |
| Hard         | 0.3441    | 0.2726    | 0.8357     | 0.2830     | 0.5835     | 1.0159     | 0.2202     |
| Average      | 0.3858    | 0.3199    | 0.7982     | 0.2779     | 0.5256     | 0.9342     | 0.2112     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2969    | 0.1913    | 0.8530     | 0.2929     | 0.5833     | 1.1048     | 0.2582     |
| Moderate     | 0.2982    | 0.1908    | 0.8504     | 0.2929     | 0.5734     | 1.1158     | 0.2552     |
| Hard         | 0.2421    | 0.1201    | 0.8815     | 0.3257     | 0.6906     | 1.2062     | 0.2822     |
| Average      | 0.2791    | 0.1674    | 0.8616     | 0.3038     | 0.6158     | 1.1423     | 0.2652     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3956    | 0.3151    | 0.8094     | 0.2757     | 0.4726     | 0.8566     | 0.2052     |
| Moderate     | 0.3686    | 0.2812    | 0.8331     | 0.2785     | 0.5047     | 0.8900     | 0.2131     |
| Hard         | 0.3468    | 0.2482    | 0.8585     | 0.2793     | 0.5173     | 0.9033     | 0.2149     |
| Average      | 0.3703    | 0.2815    | 0.8337     | 0.2778     | 0.4982     | 0.8833     | 0.2111     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3494    | 0.2715    | 0.8262     | 0.2833     | 0.5439     | 0.9719     | 0.2380     |
| Moderate     | 0.2312    | 0.1232    | 0.9234     | 0.3179     | 0.7635     | 1.1908     | 0.2995     |
| Hard         | 0.2090    | 0.1012    | 0.9443     | 0.3401     | 0.8029     | 1.2316     | 0.3283     |
| Average      | 0.2632    | 0.1653    | 0.8980     | 0.3138     | 0.7034     | 1.1314     | 0.2886     |



## References
```bib
@article{liu2022petr,
  title = {PETR: Position Embedding Transformation for Multi-View 3D Object Detection},
  author = {Liu, Yingfei and Wang, Tiancai and Zhang, Xiangyu and Sun, Jian},
  journal = {arXiv preprint arXiv:2203.05625},
  year = {2022},
}
```
