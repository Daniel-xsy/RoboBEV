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


## DETR3D

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean       | 0.4224 | 0.3468 | 0.7647 | 0.2678 | 0.3917 | 0.8754 | 0.2108 |
| |
| Cam Crash   | 0.2859    | 0.1144    | 0.8400     | 0.2821     | 0.4707     | 0.8992     | 0.2202     |
| Frame Lost  | 0.2604    | 0.0898    | 0.8647     | 0.3030     | 0.5041     | 0.9297     | 0.2439     |
| Color Quant | 0.3177    | 0.2165    | 0.8953     | 0.2816     | 0.5266     | 0.9813     | 0.2483     |
| Motion Blur | 0.2661    | 0.1479    | 0.9146     | 0.3085     | 0.6351     | 1.0385     | 0.2526     |
| Brightness  | 0.4002    | 0.3149    | 0.7915     | 0.2703     | 0.4348     | 0.8733     | 0.2028     |
| Low Light   | 0.2786    | 0.1559    | 0.8768     | 0.2947     | 0.5802     | 1.0290     | 0.2654     |
| Fog         | 0.3912    | 0.3007    | 0.7961     | 0.2711     | 0.4326     | 0.8807     | 0.2110     |
| Snow        | 0.1913    | 0.0776    | 0.9714     | 0.3752     | 0.7486     | 1.2478     | 0.3797     |


## Experiment Log

> Time: Sun Feb 12 14:09:13 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3284    | 0.1760    | 0.8120     | 0.2769     | 0.4190     | 0.8741     | 0.2138     |
| Moderate     | 0.2594    | 0.0776    | 0.8709     | 0.2821     | 0.4674     | 0.9513     | 0.2225     |
| Hard         | 0.2700    | 0.0895    | 0.8371     | 0.2874     | 0.5258     | 0.8723     | 0.2244     |
| Average      | 0.2859    | 0.1144    | 0.8400     | 0.2821     | 0.4707     | 0.8992     | 0.2202     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3437    | 0.2004    | 0.7959     | 0.2734     | 0.4139     | 0.8753     | 0.2063     |
| Moderate     | 0.2514    | 0.0600    | 0.8680     | 0.2880     | 0.4980     | 0.9172     | 0.2145     |
| Hard         | 0.1860    | 0.0092    | 0.9302     | 0.3476     | 0.6005     | 0.9965     | 0.3110     |
| Average      | 0.2604    | 0.0898    | 0.8647     | 0.3030     | 0.5041     | 0.9297     | 0.2439     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3990    | 0.3169    | 0.8051     | 0.2678     | 0.4159     | 0.8900     | 0.2163     |
| Moderate     | 0.3255    | 0.2277    | 0.8831     | 0.2764     | 0.4983     | 0.9717     | 0.2544     |
| Hard         | 0.2287    | 0.1049    | 0.9978     | 0.3005     | 0.6657     | 1.0823     | 0.2741     |
| Average      | 0.3177    | 0.2165    | 0.8953     | 0.2816     | 0.5266     | 0.9813     | 0.2483     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3721    | 0.2813    | 0.8103     | 0.2756     | 0.4653     | 0.9047     | 0.2295     |
| Moderate     | 0.2343    | 0.1023    | 0.9436     | 0.3137     | 0.6570     | 1.0311     | 0.2545     |
| Hard         | 0.1918    | 0.0601    | 0.9900     | 0.3363     | 0.7829     | 1.1796     | 0.2738     |
| Average      | 0.2661    | 0.1479    | 0.9146     | 0.3085     | 0.6351     | 1.0385     | 0.2526     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4225    | 0.3411    | 0.7668     | 0.2680     | 0.3981     | 0.8427     | 0.2052     |
| Moderate     | 0.4012    | 0.3161    | 0.7893     | 0.2702     | 0.4301     | 0.8764     | 0.2025     |
| Hard         | 0.3769    | 0.2876    | 0.8183     | 0.2728     | 0.4763     | 0.9008     | 0.2006     |
| Average      | 0.4002    | 0.3149    | 0.7915     | 0.2703     | 0.4348     | 0.8733     | 0.2028     |


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
| Easy         | 0.4047    | 0.3161    | 0.7841     | 0.2700     | 0.4216     | 0.8545     | 0.2037     |
| Moderate     | 0.3922    | 0.3026    | 0.7923     | 0.2701     | 0.4334     | 0.8813     | 0.2139     |
| Hard         | 0.3767    | 0.2833    | 0.8118     | 0.2731     | 0.4428     | 0.9064     | 0.2153     |
| Average      | 0.3912    | 0.3007    | 0.7961     | 0.2711     | 0.4326     | 0.8807     | 0.2110     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2432    | 0.1212    | 0.9459     | 0.3114     | 0.6199     | 1.1665     | 0.2973     |
| Moderate     | 0.1692    | 0.0602    | 0.9792     | 0.4073     | 0.8043     | 1.2775     | 0.4185     |
| Hard         | 0.1616    | 0.0514    | 0.9890     | 0.4068     | 0.8215     | 1.2994     | 0.4233     |
| Average      | 0.1913    | 0.0776    | 0.9714     | 0.3752     | 0.7486     | 1.2478     | 0.3797     |



## References

```bib
@inproceedings{wang2021detr3d,
   title = {DETR3D: 3D Object Detection from Multi-View Images via 3D-to-2D Queries},
   author = {Wang, Yue and Guizilini, Vitor and Zhang, Tianyuan and Wang, Yilun and Zhao, Hang and and Solomon, Justin M.},
   booktitle = {The Conference on Robot Learning},
   year = {2021},
}
```
