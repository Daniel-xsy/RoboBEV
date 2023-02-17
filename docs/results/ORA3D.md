<img src="F:\Research\Robust BEV Detection\Robust-BEV-Detection\docs\figs\logo2.png" align="right" width="30%">

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


## ORA3D-r101

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|                |         |         |          |          |          |          |          |
|     Clean      | 0.4436 | 0.3677 | 0.7319 | 0.2698 | 0.3890 | 0.8150 | 0.1975 |
|                |         |         |          |          |          |          |          |
|   Cam Crash    | 0.3055    | 0.1275    | 0.7952     | 0.2803     | 0.4549     | 0.8376     | 0.2145     |
|   Frame Lost   | 0.2750    | 0.0997    | 0.8362     | 0.3075     | 0.4963     | 0.8747     | 0.2340     |
|  Color Quant   | 0.3360    | 0.2382    | 0.8479     | 0.2848     | 0.5249     | 0.9516     | 0.2432     |
|  Motion Blur   | 0.2647    | 0.1527    | 0.8656     | 0.3497     | 0.6251     | 1.0433     | 0.3160     |
|   Brightness   | 0.4075    | 0.3252    | 0.7740     | 0.2741     | 0.4620     | 0.8372     | 0.2029     |
|   Low Light    | 0.3088    | 0.2047    | 0.8259     | 0.2907     | 0.5661     | 1.0612     | 0.2528     |
|      Fog       | 0.3959    | 0.3084    | 0.7822     | 0.2753     | 0.4515     | 0.8685     | 0.2048     |
|      Snow      | 0.1898    | 0.0757    | 0.9404     | 0.3857     | 0.7665     | 1.2890     | 0.3879     |


## Experiment Log

> Time:Time: Thu Jan 26 15:18:22 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3506    | 0.1950    | 0.7696     | 0.2739     | 0.4119     | 0.8088     | 0.2046     |
|   Moderate   | 0.2794    | 0.0882    | 0.8228     | 0.2826     | 0.4428     | 0.8784     | 0.2205     |
|     Hard     | 0.2865    | 0.0994    | 0.7933     | 0.2845     | 0.5101     | 0.8255     | 0.2183     |
|   Average    | 0.3055    | 0.1275    | 0.7952     | 0.2803     | 0.4549     | 0.8376     | 0.2145     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3594    | 0.2174    | 0.7673     | 0.2743     | 0.4226     | 0.8302     | 0.1985     |
|   Moderate   | 0.2653    | 0.0696    | 0.8288     | 0.2904     | 0.4953     | 0.8713     | 0.2097     |
|     Hard     | 0.2003    | 0.0121    | 0.9124     | 0.3577     | 0.5711     | 0.9225     | 0.2937     |
|   Average    | 0.2750    | 0.0997    | 0.8362     | 0.3075     | 0.4963     | 0.8747     | 0.2340     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4194    | 0.3428    | 0.7709     | 0.2694     | 0.4063     | 0.8611     | 0.2117     |
|   Moderate   | 0.3496    | 0.2534    | 0.8246     | 0.2775     | 0.5054     | 0.9301     | 0.2331     |
|     Hard     | 0.2389    | 0.1185    | 0.9481     | 0.3074     | 0.6630     | 1.0636     | 0.2849     |
|   Average    | 0.3360    | 0.2382    | 0.8479     | 0.2848     | 0.5249     | 0.9516     | 0.2432     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3844    | 0.2961    | 0.7830     | 0.2799     | 0.4802     | 0.8804     | 0.2134     |
|   Moderate   | 0.2348    | 0.1021    | 0.9026     | 0.3154     | 0.6737     | 1.0273     | 0.2714     |
|     Hard     | 0.1750    | 0.0599    | 0.9111     | 0.4538     | 0.7214     | 1.2221     | 0.4632     |
|   Average    | 0.2647    | 0.1527    | 0.8656     | 0.3497     | 0.6251     | 1.0433     | 0.3160     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4398    | 0.3612    | 0.7458     | 0.2691     | 0.3976     | 0.7939     | 0.2019     |
|   Moderate   | 0.4072    | 0.3261    | 0.7735     | 0.2728     | 0.4613     | 0.8455     | 0.2050     |
|     Hard     | 0.3756    | 0.2882    | 0.8028     | 0.2805     | 0.5272     | 0.8721     | 0.2019     |
|   Average    | 0.4075    | 0.3252    | 0.7740     | 0.2741     | 0.4620     | 0.8372     | 0.2029     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.3257    | 0.2292    | 0.8094     | 0.2844     | 0.5523     | 1.0198     | 0.2428     |
|   Moderate   | 0.3268    | 0.2291    | 0.8084     | 0.2836     | 0.5404     | 1.0105     | 0.2445     |
|     Hard     | 0.2739    | 0.1559    | 0.8598     | 0.3042     | 0.6056     | 1.1532     | 0.2710     |
|   Average    | 0.3088    | 0.2047    | 0.8259     | 0.2907     | 0.5661     | 1.0612     | 0.2528     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.4157    | 0.3308    | 0.7635     | 0.2717     | 0.4238     | 0.8362     | 0.2022     |
|   Moderate   | 0.3982    | 0.3125    | 0.7831     | 0.2750     | 0.4508     | 0.8667     | 0.2048     |
|     Hard     | 0.3740    | 0.2819    | 0.7999     | 0.2793     | 0.4800     | 0.9027     | 0.2075     |
|   Average    | 0.3959    | 0.3084    | 0.7822     | 0.2753     | 0.4515     | 0.8685     | 0.2048     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
|              |         |         |          |          |          |          |          |
|     Easy     | 0.2457    | 0.1197    | 0.9051     | 0.3191     | 0.6306     | 1.2343     | 0.2871     |
|   Moderate   | 0.1668    | 0.0584    | 0.9440     | 0.4163     | 0.8276     | 1.2921     | 0.4359     |
|     Hard     | 0.1570    | 0.0492    | 0.9721     | 0.4217     | 0.8413     | 1.3406     | 0.4408     |
|   Average    | 0.1898    | 0.0757    | 0.9404     | 0.3857     | 0.7665     | 1.2890     | 0.3879     |



## References

```bib
@article{roh2022ora3d,
  title={Ora3d: Overlap region aware multi-view 3d object detection},
  author={Roh, Wonseok and Chang, Gyusam and Moon, Seokha and Nam, Giljoo and Kim, Chanyoung and Kim, Younghyun and Kim, Sangpil and Kim, Jinkyu},
  journal={arXiv preprint arXiv:2207.00865},
  year={2022}
}
```