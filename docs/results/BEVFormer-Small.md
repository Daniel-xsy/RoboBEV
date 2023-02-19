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


## BEVFormer-Small

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.4787  | 0.3700  | 0.7212   | 0.2792   | 0.4065   | 0.4364   | 0.2201   |
| |
| Cam Crash      | 0.2771  | 0.1130  | 0.8627   | 0.3099   | 0.5398   | 0.8376   | 0.2446   |
| Frame Lost     | 0.2459  | 0.0933  | 0.8959   | 0.3411   | 0.5742   | 0.9154   | 0.2804   |
| Color Quant    | 0.3275  | 0.2109  | 0.8476   | 0.2943   | 0.5234   | 0.8539   | 0.2601   |
| Motion Blur    | 0.2570  | 0.1344  | 0.8995   | 0.3264   | 0.6774   | 0.9625   | 0.2605   |
| Brightness     | 0.3741  | 0.2697  | 0.8064   | 0.2830   | 0.4796   | 0.8162   | 0.2226   |
| Low Light      | 0.2413    | 0.1191    | 0.8838     | 0.3598     | 0.6470     | 1.0391     | 0.3323     |
| Fog            | 0.3583  | 0.2486  | 0.8131   | 0.2862   | 0.5056   | 0.8301   | 0.2251   |
| Snow           | 0.1809  | 0.0635  | 0.9630   | 0.3855   | 0.7741   | 1.1002   | 0.3863   |


## Experiment Log

> Time: Mon Feb 13 13:47:14 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3295  | 0.1801  | 0.8284   | 0.2943   | 0.4946   | 0.7597   | 0.2285   |
| Moderate     | 0.2664  | 0.0965  | 0.8986   | 0.3087   | 0.5365   | 0.8226   | 0.2524   |
| Hard         | 0.2353  | 0.0625  | 0.8611   | 0.3266   | 0.5884   | 0.9304   | 0.2530   |
| Average      | 0.2771  | 0.1130  | 0.8627   | 0.3099   | 0.5398   | 0.8376   | 0.2446   |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3283  | 0.1947  | 0.8296   | 0.2923   | 0.4934   | 0.8405   | 0.2350   |
| Moderate     | 0.2378  | 0.0684  | 0.9013   | 0.3229   | 0.5732   | 0.9090   | 0.2576   |
| Hard         | 0.1717  | 0.0167  | 0.9569   | 0.4081   | 0.6559   | 0.9968   | 0.3486   |
| Average      | 0.2459  | 0.0933  | 0.8959   | 0.3411   | 0.5742   | 0.9154   | 0.2804   |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3896  | 0.2884  | 0.7960   | 0.2806   | 0.4468   | 0.7878   | 0.2345   |
| Moderate     | 0.3415  | 0.2247  | 0.8281   | 0.2868   | 0.5023   | 0.8339   | 0.2578   |
| Hard         | 0.2515  | 0.1197  | 0.9186   | 0.3156   | 0.6211   | 0.9401   | 0.2881   |
| Average      | 0.3275  | 0.2109  | 0.8476   | 0.2943   | 0.5234   | 0.8539   | 0.2601   |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3582  | 0.2465  | 0.8195   | 0.2883   | 0.4981   | 0.8146   | 0.2304   |
| Moderate     | 0.2246  | 0.0970  | 0.9206   | 0.3333   | 0.7192   | 1.0316   | 0.2657   |
| Hard         | 0.1883  | 0.0597  | 0.9583   | 0.3575   | 0.8148   | 1.0413   | 0.2853   |
| Average      | 0.2570  | 0.1344  | 0.8995   | 0.3264   | 0.6774   | 0.9625   | 0.2605   |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3936  | 0.2956  | 0.7911   | 0.2807   | 0.4517   | 0.7910   | 0.2273   |
| Moderate     | 0.3735  | 0.2690  | 0.8093   | 0.2844   | 0.4798   | 0.8132   | 0.2237   |
| Hard         | 0.3551  | 0.2446  | 0.8188   | 0.2840   | 0.5073   | 0.8445   | 0.2168   |
| Average      | 0.3741  | 0.2697  | 0.8064   | 0.2830   | 0.4796   | 0.8162   | 0.2226   |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3043    | 0.1808    | 0.8480     | 0.2998     | 0.5862     | 0.8789     | 0.2488     |
| Moderate     | 0.2449    | 0.1198    | 0.9006     | 0.3235     | 0.6576     | 1.0033     | 0.2687     |
| Hard         | 0.1748    | 0.0567    | 0.9028     | 0.4560     | 0.6972     | 1.2352     | 0.4795     |
| Average      | 0.2413    | 0.1191    | 0.8838     | 0.3598     | 0.6470     | 1.0391     | 0.3323     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3711  | 0.2650  | 0.8033   | 0.2837   | 0.4920   | 0.8171   | 0.2176   |
| Moderate     | 0.3604  | 0.2511  | 0.8082   | 0.2857   | 0.5050   | 0.8275   | 0.2246   |
| Hard         | 0.3433  | 0.2298  | 0.8279   | 0.2893   | 0.5197   | 0.8458   | 0.2332   |
| Average      | 0.3583  | 0.2486  | 0.8131   | 0.2862   | 0.5056   | 0.8301   | 0.2251   |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2212  | 0.0951  | 0.9511   | 0.3311   | 0.6783   | 1.0630   | 0.3032   |
| Moderate     | 0.1648  | 0.0509  | 0.9654   | 0.4098   | 0.8067   | 1.0791   | 0.4246   |
| Hard         | 0.1567  | 0.0446  | 0.9724   | 0.4155   | 0.8374   | 1.1585   | 0.4310   |
| Average      | 0.1809  | 0.0635  | 0.9630   | 0.3855   | 0.7741   | 1.1002   | 0.3863   |



## References

```bib
@article{li2022bevformer,
  title = {BEVFormer: Learning Bird’s-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers},
  author = {Li, Zhiqi and Wang, Wenhai and Li, Hongyang and Xie, Enze and Sima, Chonghao and Lu, Tong and Qiao, Yu and Dai, Jifeng},
  journal = {arXiv preprint arXiv:2203.17270},
  year = {2022},
}
```
