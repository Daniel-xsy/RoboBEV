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


## SOLOFusion CBGS

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.5381    | 0.4299    | 0.5842     | 0.2747     | 0.4564     | 0.2426     | 0.2103     |
| |
| Cam Crash      | 0.3806    | 0.1590    | 0.6607     | 0.2773     | 0.5186     | 0.3152     | 0.2176     |
| Frame Lost     | 0.3464    | 0.1671    | 0.7161     | 0.3042     | 0.5557     | 0.5292     | 0.2668     |
| Color Quant    | 0.4058    | 0.2572    | 0.6910     | 0.3200     | 0.6217     | 0.3434     | 0.2514     |
| Motion Blur    | 0.3642    | 0.2019    | 0.7191     | 0.3244     | 0.6643     | 0.3834     | 0.2762     |
| Brightness     | 0.4329    | 0.2959    | 0.6532     | 0.3238     | 0.5353     | 0.3808     | 0.2577     |
| Low Light      | 0.2626    | 0.1237    | 0.7258     | 0.4567     | 0.7598     | 0.5910     | 0.4597     |
| Fog            | 0.4480    | 0.2923    | 0.6502     | 0.2883     | 0.5496     | 0.2958     | 0.1973     |
| Snow           | 0.1376    | 0.0561    | 0.8722     | 0.6480     | 0.8219     | 0.8363     | 0.7255     |


## Experiment Log

> Time: Thu Apr  6 15:32:35 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4354    | 0.2507    | 0.6388     | 0.2746     | 0.4974     | 0.2833     | 0.2050     |
| Moderate     | 0.3589    | 0.1192    | 0.7112     | 0.2740     | 0.4860     | 0.3153     | 0.2203     |
| Hard         | 0.3474    | 0.1073    | 0.6322     | 0.2833     | 0.5724     | 0.3470     | 0.2275     |
| Average      | 0.3806    | 0.1590    | 0.6607     | 0.2773     | 0.5186     | 0.3152     | 0.2176     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4695    | 0.3199    | 0.6321     | 0.2771     | 0.4718     | 0.3071     | 0.2166     |
| Moderate     | 0.3349    | 0.1381    | 0.7287     | 0.2907     | 0.5627     | 0.5172     | 0.2427     |
| Hard         | 0.2348    | 0.0434    | 0.7874     | 0.3449     | 0.6325     | 0.7632     | 0.3410     |
| Average      | 0.3464    | 0.1671    | 0.7161     | 0.3042     | 0.5557     | 0.5292     | 0.2668     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.5153    | 0.3935    | 0.6014     | 0.2793     | 0.4698     | 0.2545     | 0.2095     |
| Moderate     | 0.4339    | 0.2723    | 0.6643     | 0.2910     | 0.5546     | 0.2960     | 0.2170     |
| Hard         | 0.2683    | 0.1057    | 0.8073     | 0.3897     | 0.8408     | 0.4796     | 0.3278     |
| Average      | 0.4058    | 0.2572    | 0.6910     | 0.3200     | 0.6217     | 0.3434     | 0.2514     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4856    | 0.3532    | 0.6204     | 0.2860     | 0.5388     | 0.2626     | 0.2020     |
| Moderate     | 0.3410    | 0.1527    | 0.7529     | 0.3018     | 0.6926     | 0.3655     | 0.2411     |
| Hard         | 0.2661    | 0.0999    | 0.7841     | 0.3854     | 0.7615     | 0.5222     | 0.3855     |
| Average      | 0.3642    | 0.2019    | 0.7191     | 0.3244     | 0.6643     | 0.3834     | 0.2762     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.5035    | 0.3723    | 0.6119     | 0.2795     | 0.4692     | 0.2770     | 0.1891     |
| Moderate     | 0.4451    | 0.2871    | 0.6555     | 0.2816     | 0.5303     | 0.3281     | 0.1890     |
| Hard         | 0.3500    | 0.2282    | 0.6923     | 0.4103     | 0.6063     | 0.5374     | 0.3951     |
| Average      | 0.4329    | 0.2959    | 0.6532     | 0.3238     | 0.5353     | 0.3808     | 0.2577     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3339    | 0.1807    | 0.6933     | 0.3743     | 0.7006     | 0.4598     | 0.3370     |
| Moderate     | 0.2732    | 0.1287    | 0.6953     | 0.4181     | 0.7404     | 0.6177     | 0.4400     |
| Hard         | 0.1806    | 0.0618    | 0.7889     | 0.5777     | 0.8383     | 0.6955     | 0.6022     |
| Average      | 0.2626    | 0.1237    | 0.7258     | 0.4567     | 0.7598     | 0.5910     | 0.4597     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.4656    | 0.3199    | 0.6426     | 0.2825     | 0.5436     | 0.2817     | 0.1932     |
| Moderate     | 0.4505    | 0.2952    | 0.6465     | 0.2878     | 0.5426     | 0.2970     | 0.1972     |
| Hard         | 0.4280    | 0.2618    | 0.6614     | 0.2946     | 0.5626     | 0.3088     | 0.2014     |
| Average      | 0.4480    | 0.2923    | 0.6502     | 0.2883     | 0.5496     | 0.2958     | 0.1973     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2638    | 0.1110    | 0.7757     | 0.4273     | 0.6741     | 0.5913     | 0.4489     |
| Moderate     | 0.0685    | 0.0301    | 0.9115     | 0.7930     | 0.8853     | 0.9709     | 0.9051     |
| Hard         | 0.0807    | 0.0272    | 0.9295     | 0.7238     | 0.9064     | 0.9468     | 0.8226     |
| Average      | 0.1376    | 0.0561    | 0.8722     | 0.6480     | 0.8219     | 0.8363     | 0.7255     |



## References

```bib
@article{Park2022TimeWT,
  title={Time Will Tell: New Outlooks and A Baseline for Temporal Multi-View 3D Object Detection},
  author={Park, Jinhyung and Xu, Chenfeng and Yang, Shijia and Keutzer, Kurt and Kitani, Kris and Tomizuka, Masayoshi and Zhan, Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
