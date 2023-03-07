<img src="..\figs\logo2.png" align="right" width="30%">

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


## BEVDet-r101-fcos3d-pretrain

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.3780 | 0.2846 | 0.7274 | 0.2796 | 0.5517 | 0.8581 | 0.2264 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.2442    | 0.0928    | 0.8020     | 0.3384     | 0.5815     | 1.0285     | 0.3453     |
|   Frame Lost   | 0.1962    | 0.0720    | 0.8320     | 0.4427     | 0.6830     | 1.0063     | 0.4684     |
|  Color Quant   | 0.3041    | 0.2064    | 0.7815     | 0.3247     | 0.6251     | 0.9955     | 0.3212     |
|  Motion Blur   | 0.2590    | 0.1512    | 0.7826     | 0.3675     | 0.6412     | 1.1481     | 0.3973     |
|   Brightness   | 0.2599    | 0.1714    | 0.7910     | 0.3963     | 0.6828     | 1.1539     | 0.4242     |
|   Low Light    | 0.1393    | 0.0613    | 0.8761     | 0.5631     | 0.8235     | 1.1739     | 0.6510     |
|      Fog       | 0.2073    | 0.0984    | 0.8521     | 0.4107     | 0.6897     | 1.2659     | 0.4668     |
|      Snow      | 0.0939    | 0.0301    | 0.9494     | 0.6685     | 0.8397     | 1.2412     | 0.7535     |


## Experiment Log

> Time: Mon Mar  6 22:08:01 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3036    | 0.1480    | 0.7628     | 0.2800     | 0.5572     | 0.8635     | 0.2410     |
|   Moderate   | 0.2253    | 0.0629    | 0.8272     | 0.3323     | 0.5746     | 1.0089     | 0.3277     |
|     Hard     | 0.2039    | 0.0675    | 0.8161     | 0.4030     | 0.6128     | 1.2132     | 0.4672     |
|   Average    | 0.2442    | 0.0928    | 0.8020     | 0.3384     | 0.5815     | 1.0285     | 0.3453     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3058    | 0.1651    | 0.7656     | 0.2813     | 0.5715     | 0.9164     | 0.2329     |
|   Moderate   | 0.2142    | 0.0449    | 0.8123     | 0.3347     | 0.6042     | 1.0881     | 0.3317     |
|     Hard     | 0.0685    | 0.0059    | 0.9182     | 0.7122     | 0.8732     | 1.0145     | 0.8407     |
|   Average    | 0.1962    | 0.0720    | 0.8320     | 0.4427     | 0.6830     | 1.0063     | 0.4684     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3726    | 0.2790    | 0.7226     | 0.2784     | 0.5556     | 0.8760     | 0.2365     |
|   Moderate   | 0.3299    | 0.2296    | 0.7658     | 0.2799     | 0.6044     | 0.9406     | 0.2579     |
|     Hard     | 0.2096    | 0.1105    | 0.8561     | 0.4157     | 0.7154     | 1.1700     | 0.4691     |
|   Average    | 0.3041    | 0.2064    | 0.7815     | 0.3247     | 0.6251     | 0.9955     | 0.3212     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3504    | 0.2502    | 0.7417     | 0.2795     | 0.5642     | 0.9306     | 0.2308     |
|   Moderate   | 0.2498    | 0.1242    | 0.7825     | 0.3365     | 0.6505     | 1.1741     | 0.3532     |
|     Hard     | 0.1768    | 0.0790    | 0.8235     | 0.4864     | 0.7090     | 1.3395     | 0.6078     |
|   Average    | 0.2590    | 0.1512    | 0.7826     | 0.3675     | 0.6412     | 1.1481     | 0.3973     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3539    | 0.2532    | 0.7359     | 0.2794     | 0.5774     | 0.8897     | 0.2445     |
|   Moderate   | 0.2445    | 0.1607    | 0.7900     | 0.4148     | 0.7054     | 1.2555     | 0.4482     |
|     Hard     | 0.1813    | 0.1001    | 0.8470     | 0.4947     | 0.7657     | 1.3165     | 0.5799     |
|   Average    | 0.2599    | 0.1714    | 0.7910     | 0.3963     | 0.6828     | 1.1539     | 0.4242     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2143    | 0.1077    | 0.8034     | 0.4097     | 0.7081     | 1.2486     | 0.4745     |
|   Moderate   | 0.1588    | 0.0602    | 0.8594     | 0.4910     | 0.7623     | 1.1715     | 0.5996     |
|     Hard     | 0.0447    | 0.0160    | 0.9656     | 0.7886     | 1.0000     | 1.1016     | 0.8789     |
|   Average    | 0.1393    | 0.0613    | 0.8761     | 0.5631     | 0.8235     | 1.1739     | 0.6510     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2546    | 0.1356    | 0.8267     | 0.3305     | 0.6306     | 1.1996     | 0.3441     |
|   Moderate   | 0.2072    | 0.0972    | 0.8393     | 0.4128     | 0.6949     | 1.3338     | 0.4669     |
|     Hard     | 0.1600    | 0.0625    | 0.8902     | 0.4888     | 0.7437     | 1.2643     | 0.5895     |
|   Average    | 0.2073    | 0.0984    | 0.8521     | 0.4107     | 0.6897     | 1.2659     | 0.4668     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.1691    | 0.0543    | 0.8868     | 0.4269     | 0.7774     | 1.4151     | 0.4896     |
|   Moderate   | 0.0804    | 0.0275    | 0.9744     | 0.7199     | 0.7926     | 1.1624     | 0.8465     |
|     Hard     | 0.0323    | 0.0085    | 0.9870     | 0.8587     | 0.9490     | 1.1461     | 0.9245     |
|   Average    | 0.0939    | 0.0301    | 0.9494     | 0.6685     | 0.8397     | 1.2412     | 0.7535     |



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