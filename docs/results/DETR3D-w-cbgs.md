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


## DETR3D w/ cbgs


| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean       | 0.4341    | 0.3494    | 0.7163     | 0.2682     | 0.3798     | 0.8421     | 0.1997     |
| |
| Cam Crash   | 0.2991    | 0.1174    | 0.7932     | 0.2853     | 0.4575     | 0.8471     | 0.2131     |
| Frame Lost  | 0.2685    | 0.0923    | 0.8268     | 0.3135     | 0.5042     | 0.8867     | 0.2455     |
| Color Quant | 0.3235    | 0.2152    | 0.8571     | 0.2875     | 0.5350     | 0.9354     | 0.2400     |
| Motion Blur | 0.2542    | 0.1385    | 0.8909     | 0.3355     | 0.6707     | 1.0682     | 0.2928     |
| Brightness  | 0.4154    | 0.3200    | 0.7357     | 0.2720     | 0.4086     | 0.8302     | 0.1990     |
| Low Light   | 0.2786    | 0.1559    | 0.8768     | 0.2947     | 0.5802     | 1.0290     | 0.2654     |
| Fog         | 0.4020    | 0.3012    | 0.7552     | 0.2710     | 0.4237     | 0.8302     | 0.2054     |
| Snow        | 0.1925    | 0.0702    | 0.9246     | 0.3793     | 0.7648     | 1.2585     | 0.3577     |


## Experiment Log

> Time: Sun Feb 12 09:52:47 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3399    | 0.1790    | 0.7659     | 0.2797     | 0.4204     | 0.8238     | 0.2064     |
| Moderate     | 0.2733    | 0.0812    | 0.8138     | 0.2851     | 0.4587     | 0.8909     | 0.2243     |
| Hard         | 0.2840    | 0.0920    | 0.7998     | 0.2910     | 0.4935     | 0.8266     | 0.2085     |
| Average      | 0.2991    | 0.1174    | 0.7932     | 0.2853     | 0.4575     | 0.8471     | 0.2131     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3554    | 0.2044    | 0.7517     | 0.2749     | 0.4039     | 0.8415     | 0.1965     |
| Moderate     | 0.2606    | 0.0626    | 0.8209     | 0.2942     | 0.4963     | 0.8785     | 0.2171     |
| Hard         | 0.1894    | 0.0098    | 0.9077     | 0.3713     | 0.6125     | 0.9401     | 0.3229     |
| Average      | 0.2685    | 0.0923    | 0.8268     | 0.3135     | 0.5042     | 0.8867     | 0.2455     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4133    | 0.3226    | 0.7530     | 0.2692     | 0.3985     | 0.8464     | 0.2124     |
| Moderate     | 0.3396    | 0.2307    | 0.8498     | 0.2804     | 0.4733     | 0.9189     | 0.2355     |
| Hard         | 0.2175    | 0.0925    | 0.9685     | 0.3130     | 0.7333     | 1.0408     | 0.2722     |
| Average      | 0.3235    | 0.2152    | 0.8571     | 0.2875     | 0.5350     | 0.9354     | 0.2400     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3758    | 0.2749    | 0.7891     | 0.2762     | 0.4583     | 0.8815     | 0.2115     |
| Moderate     | 0.2218    | 0.0896    | 0.9283     | 0.3197     | 0.7191     | 1.1042     | 0.2629     |
| Hard         | 0.1650    | 0.0508    | 0.9554     | 0.4105     | 0.8346     | 1.2189     | 0.4040     |
| Average      | 0.2542    | 0.1385    | 0.8909     | 0.3355     | 0.6707     | 1.0682     | 0.2928     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4341    | 0.3440    | 0.7121     | 0.2698     | 0.3884     | 0.8130     | 0.1959     |
| Moderate     | 0.4181    | 0.3232    | 0.7334     | 0.2716     | 0.4057     | 0.8255     | 0.1984     |
| Hard         | 0.3940    | 0.2926    | 0.7617     | 0.2746     | 0.4318     | 0.8522     | 0.2026     |
| Average      | 0.4154    | 0.3200    | 0.7357     | 0.2720     | 0.4086     | 0.8302     | 0.1990     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3334    | 0.2231    | 0.8313     | 0.2778     | 0.4988     | 0.9291     | 0.2441     |
| Moderate     | 0.2832    | 0.1607    | 0.8551     | 0.2891     | 0.5665     | 1.0100     | 0.2613     |
| Hard         | 0.2192    | 0.0839    | 0.9439     | 0.3172     | 0.6753     | 1.1478     | 0.2909     |
| Average      | 0.2786    | 0.1559    | 0.8768     | 0.2947     | 0.5802     | 1.0290     | 0.2654     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4181    | 0.3199    | 0.7345     | 0.2692     | 0.3974     | 0.8136     | 0.2033     |
| Moderate     | 0.4053    | 0.3045    | 0.7508     | 0.2704     | 0.4160     | 0.8281     | 0.2038     |
| Hard         | 0.3826    | 0.2791    | 0.7802     | 0.2733     | 0.4578     | 0.8490     | 0.2092     |
| Average      | 0.4020    | 0.3012    | 0.7552     | 0.2710     | 0.4237     | 0.8302     | 0.2054     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2527    | 0.1262    | 0.8959     | 0.3077     | 0.6333     | 1.1899     | 0.2672     |
| Moderate     | 0.1718    | 0.0503    | 0.9228     | 0.4092     | 0.8054     | 1.2890     | 0.3964     |
| Hard         | 0.1530    | 0.0342    | 0.9550     | 0.4210     | 0.8556     | 1.2965     | 0.4094     |
| Average      | 0.1925    | 0.0702    | 0.9246     | 0.3793     | 0.7648     | 1.2585     | 0.3577     |



## References

```bib
@inproceedings{wang2021detr3d,
   title = {DETR3D: 3D Object Detection from Multi-View Images via 3D-to-2D Queries},
   author = {Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle = {The Conference on Robot Learning},
   year = {2021},
}
```

