<img src="../figs/logo2.png" align="right" width="30%">

# RoboBEV Benchmark

The official [nuScenes metric](https://www.nuscenes.org/object-detection/?externalData=all&mapData=all&modalities=Any) are considered in our benchmark:

### Average Precision (AP)

The average precision (AP) defines a match by thresholding the 2D center distance d on the ground plane instead of the intersection over union (IoU). This is done in order to decouple detection from object size and orientation but also because objects with small footprints, like pedestrians and bikes, if detected with a small translation error, give 0 IoU.
We then calculate AP as the normalized area under the precision-recall curve for recall and precision over 10%. Operating points where recall or precision is less than 10% are removed in order to minimize the impact of noise commonly seen in low precision and recall regions. If no operating point in this region is achieved, the AP for that class is set to zero. We then average over-matching thresholds of $\mathbb{D}=\{0.5, 1, 2, 4\}$ meters and the set of classes $\mathbb{C}$ :

$$
\text{mAP}= \frac{1}{|\mathbb{C}||\mathbb{D}|}\sum_{c\in\mathbb{C}}\sum_{d\in\mathbb{D}}\text{AP}_{c,d}
$$

### True Positive (TP)

All TP metrics are calculated using $d=2$ m center distance during matching, and they are all designed to be positive scalars. Matching and scoring happen independently per class and each metric is the average of the cumulative mean at each achieved recall level above 10%. If a 10% recall is not achieved for a particular class, all TP errors for that class are set to 1. 

- **Average Translation Error (ATE)** is the Euclidean center distance in 2D (units in meters). 
- **Average Scale Error (ASE)** is the 3D intersection-over-union (IoU) after aligning orientation and translation (1 − IoU).
- **Average Orientation Error (AOE)** is the smallest yaw angle difference between prediction and ground truth (radians). All angles are measured on a full 360-degree period except for barriers where they are measured on a 180-degree period.
- **Average Velocity Error (AVE)** is the absolute velocity error as the L2 norm of the velocity differences in 2D (m/s).
- **Average Attribute Error (AAE)** is defined as 1 minus attribute classification accuracy (1 − acc).

### nuScenes Detection Score (NDS)
mAP with a threshold on IoU is perhaps the most popular metric for object detection. However, this metric can not capture all aspects of the nuScenes detection tasks, like velocity and attribute estimation. Further, it couples location, size, and orientation estimates. nuScenes proposed instead consolidating the different error types into a scalar score:

$$
\text{NDS} = \frac{1}{10} [5\text{mAP}+\sum_{\text{mTP}\in\mathbb{TP}} (1-\min(1, \text{mTP}))]
$$

## BEVFormer-Base-SingleFrame

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|||||||||
| Clean | 0.4129    | 0.3461    | 0.7549     | 0.2832     | 0.4520     | 0.8917     | 0.2194     |
|||||||||
| Cam Crash | 0.2879    | 0.1240    | 0.8041     | 0.2966     | 0.5094     | 0.8986     | 0.2323     |
| Frame Lost | 0.2642    | 0.0969    | 0.8352     | 0.3093     | 0.5748     | 0.8861     | 0.2374     |
| Color Quant | 0.3207    | 0.2243    | 0.8488     | 0.2992     | 0.5422     | 1.0003     | 0.2522     |
| Motion Blur | 0.2518    | 0.1434    | 0.8845     | 0.3248     | 0.7179     | 1.1211     | 0.2860     |
| Brightness | 0.3819    | 0.3093    | 0.7761     | 0.2861     | 0.4999     | 0.9466     | 0.2201     |
| Low Light | 0.2381    | 0.1316    | 0.8640     | 0.3602     | 0.6903     | 1.2132     | 0.3622     |
| Fog | 0.3662    | 0.2907    | 0.7938     | 0.2870     | 0.5162     | 0.9702     | 0.2254     |
| Snow | 0.1793    | 0.0687    | 0.9472     | 0.3954     | 0.8004     | 1.2524     | 0.4078     |

## Experiment Log

>Time: Sun Feb 12 10:05:19 2023

### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3328    | 0.1906    | 0.7831     | 0.2888     | 0.4590     | 0.8742     | 0.2198     |
|   Moderate   | 0.2654    | 0.0885    | 0.8346     | 0.2976     | 0.4930     | 0.9265     | 0.2368     |
|     Hard     | 0.2655    | 0.0930    | 0.7947     | 0.3035     | 0.5761     | 0.8952     | 0.2402     |
|   Average    | 0.2879    | 0.1240    | 0.8041     | 0.2966     | 0.5094     | 0.8986     | 0.2323     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3386    | 0.2094    | 0.7762     | 0.2886     | 0.4905     | 0.8862     | 0.2196     |
|   Moderate   | 0.2515    | 0.0687    | 0.8323     | 0.3057     | 0.5612     | 0.8975     | 0.2316     |
|     Hard     | 0.2023    | 0.0124    | 0.8972     | 0.3335     | 0.6726     | 0.8746     | 0.2611     |
|   Average    | 0.2642    | 0.0969    | 0.8352     | 0.3093     | 0.5748     | 0.8861     | 0.2374     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3939    | 0.3216    | 0.7708     | 0.2834     | 0.4628     | 0.9235     | 0.2279     |
|   Moderate   | 0.3299    | 0.2385    | 0.8299     | 0.2919     | 0.5330     | 0.9929     | 0.2462     |
|     Hard     | 0.2382    | 0.1127    | 0.9456     | 0.3222     | 0.6309     | 1.0844     | 0.2825     |
|   Average    | 0.3207    | 0.2243    | 0.8488     | 0.2992     | 0.5422     | 1.0003     | 0.2522     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3578    | 0.2773    | 0.7874     | 0.2899     | 0.5353     | 0.9585     | 0.2376     |
|   Moderate   | 0.2189    | 0.0969    | 0.9080     | 0.3295     | 0.7616     | 1.1423     | 0.2961     |
|     Hard     | 0.1786    | 0.0561    | 0.9580     | 0.3550     | 0.8567     | 1.2624     | 0.3244     |
|   Average    | 0.2518    | 0.1434    | 0.8845     | 0.3248     | 0.7179     | 1.1211     | 0.2860     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4089    | 0.3405    | 0.7578     | 0.2828     | 0.4657     | 0.8925     | 0.2148     |
|   Moderate   | 0.3833    | 0.3093    | 0.7726     | 0.2858     | 0.4905     | 0.9442     | 0.2200     |
|     Hard     | 0.3534    | 0.2780    | 0.7978     | 0.2897     | 0.5434     | 1.0030     | 0.2254     |
|   Average    | 0.3819    | 0.3093    | 0.7761     | 0.2861     | 0.4999     | 0.9466     | 0.2201     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2963    | 0.1965    | 0.8262     | 0.3043     | 0.6215     | 1.0934     | 0.2673     |
|   Moderate   | 0.2465    | 0.1340    | 0.8724     | 0.3243     | 0.6986     | 1.1718     | 0.3093     |
|     Hard     | 0.1716    | 0.0644    | 0.8935     | 0.4520     | 0.7509     | 1.3743     | 0.5100     |
|   Average    | 0.2381    | 0.1316    | 0.8640     | 0.3602     | 0.6903     | 1.2132     | 0.3622     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3821    | 0.3083    | 0.7808     | 0.2847     | 0.5020     | 0.9375     | 0.2150     |
|   Moderate   | 0.3690    | 0.2945    | 0.7911     | 0.2855     | 0.5087     | 0.9696     | 0.2277     |
|     Hard     | 0.3475    | 0.2693    | 0.8094     | 0.2909     | 0.5379     | 1.0034     | 0.2335     |
|   Average    | 0.3662    | 0.2907    | 0.7938     | 0.2870     | 0.5162     | 0.9702     | 0.2254     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2220    | 0.1000    | 0.9313     | 0.3396     | 0.6921     | 1.2098     | 0.3171     |
|   Moderate   | 0.1643    | 0.0577    | 0.9428     | 0.4214     | 0.8296     | 1.2669     | 0.4514     |
|     Hard     | 0.1515    | 0.0484    | 0.9675     | 0.4251     | 0.8796     | 1.2805     | 0.4549     |
|   Average    | 0.1793    | 0.0687    | 0.9472     | 0.3954     | 0.8004     | 1.2524     | 0.4078     |



## References

```bib
@article{li2022bevformer,
  title = {BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author = {Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal = {arXiv preprint arXiv:2203.17270},
  year = {2022},
}
```
