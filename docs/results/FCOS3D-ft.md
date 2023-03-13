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


## FCOS3D-fine-tune

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.3949 | 0.3214 | 0.7538 | 0.2603 | 0.4864 | 1.3321 | 0.1574 |
| |
| Cam Crash      | 0.2849    | 0.1169    | 0.7842     | 0.2693     | 0.5134     | 1.2993     | 0.1684     |
| Frame Lost     | 0.2479    | 0.0915    | 0.7912     | 0.3521     | 0.5367     | 1.3668     | 0.2989     |
| Color Quant    | 0.2574    | 0.1548    | 0.8851     | 0.3631     | 0.6378     | 1.3906     | 0.3157     |
| Motion Blur    | 0.2570    | 0.1459    | 0.8460     | 0.3318     | 0.6894     | 1.2404     | 0.2920     |
| Brightness     | 0.3218    | 0.2237    | 0.8243     | 0.2801     | 0.6179     | 1.4902     | 0.1778     |
| Low Light      | 0.1468    | 0.0491    | 0.8845     | 0.5287     | 0.7911     | 1.3388     | 0.5729     |
| Fog            | 0.3321    | 0.2319    | 0.8087     | 0.2702     | 0.5719     | 1.2989     | 0.1879     |
| Snow           | 0.1136    | 0.0448    | 0.9656     | 0.6321     | 0.7768     | 1.2827     | 0.7141     |


## Experiment Log

> Time: Sun Mar 12 21:22:28 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3250    | 0.1826    | 0.7588     | 0.2650     | 0.4817     | 1.2255     | 0.1580     |
| Moderate     | 0.2640    | 0.0816    | 0.8140     | 0.2672     | 0.5098     | 1.3355     | 0.1771     |
| Hard         | 0.2658    | 0.0864    | 0.7798     | 0.2757     | 0.5488     | 1.3369     | 0.1700     |
| Average      | 0.2849    | 0.1169    | 0.7842     | 0.2693     | 0.5134     | 1.2993     | 0.1684     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3311    | 0.1975    | 0.7625     | 0.2616     | 0.4934     | 1.3351     | 0.1592     |
| Moderate     | 0.2459    | 0.0659    | 0.7863     | 0.3214     | 0.5143     | 1.4426     | 0.2484     |
| Hard         | 0.1666    | 0.0112    | 0.8247     | 0.4734     | 0.6025     | 1.3228     | 0.4892     |
| Average      | 0.2479    | 0.0915    | 0.7912     | 0.3521     | 0.5367     | 1.3668     | 0.2989     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3666    | 0.2783    | 0.7778     | 0.2675     | 0.5084     | 1.3691     | 0.1713     |
| Moderate     | 0.2541    | 0.1450    | 0.8726     | 0.3659     | 0.6269     | 1.3820     | 0.3186     |
| Hard         | 0.1515    | 0.0412    | 1.0050     | 0.4559     | 0.7782     | 1.4207     | 0.4573     |
| Average      | 0.2574    | 0.1548    | 0.8851     | 0.3631     | 0.6378     | 1.3906     | 0.3157     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3512    | 0.2661    | 0.7846     | 0.2681     | 0.5853     | 1.2588     | 0.1807     |
| Moderate     | 0.2393    | 0.1081    | 0.8746     | 0.3022     | 0.7294     | 1.1165     | 0.2413     |
| Hard         | 0.1806    | 0.0634    | 0.8787     | 0.4252     | 0.7535     | 1.3460     | 0.4540     |
| Average      | 0.2570    | 0.1459    | 0.8460     | 0.3318     | 0.6894     | 1.2404     | 0.2920     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3667    | 0.2812    | 0.7816     | 0.2664     | 0.5388     | 1.4204     | 0.1522     |
| Moderate     | 0.3199    | 0.2186    | 0.8181     | 0.2797     | 0.6147     | 1.5017     | 0.1817     |
| Hard         | 0.2788    | 0.1711    | 0.8733     | 0.2943     | 0.7003     | 1.5484     | 0.1996     |
| Average      | 0.3218    | 0.2237    | 0.8243     | 0.2801     | 0.6179     | 1.4902     | 0.1778     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2029    | 0.0854    | 0.8532     | 0.4227     | 0.6611     | 1.4495     | 0.4604     |
| Moderate     | 0.1667    | 0.0455    | 0.8749     | 0.4396     | 0.7528     | 1.4086     | 0.4934     |
| Hard         | 0.0708    | 0.0163    | 0.9254     | 0.7239     | 0.9593     | 1.1582     | 0.7648     |
| Average      | 0.1468    | 0.0491    | 0.8845     | 0.5287     | 0.7911     | 1.3388     | 0.5729     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3479    | 0.2544    | 0.7940     | 0.2657     | 0.5534     | 1.3084     | 0.1799     |
| Moderate     | 0.3326    | 0.2330    | 0.8094     | 0.2695     | 0.5733     | 1.3030     | 0.1866     |
| Hard         | 0.3158    | 0.2085    | 0.8228     | 0.2754     | 0.5889     | 1.2852     | 0.1973     |
| Average      | 0.3321    | 0.2319    | 0.8087     | 0.2702     | 0.5719     | 1.2989     | 0.1879     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.1930    | 0.0843    | 0.9109     | 0.4422     | 0.6689     | 1.4744     | 0.4699     |
| Moderate     | 0.0744    | 0.0237    | 0.9878     | 0.7278     | 0.8236     | 1.1848     | 0.8356     |
| Hard         | 0.0733    | 0.0265    | 0.9981     | 0.7264     | 0.8379     | 1.1890     | 0.8367     |
| Average      | 0.1136    | 0.0448    | 0.9656     | 0.6321     | 0.7768     | 1.2827     | 0.7141     |



## References

```bib
@inproceedings{wang2021fcos3d,
  title={Fcos3d: Fully convolutional one-stage monocular 3d object detection},
  author={Wang, Tai and Zhu, Xinge and Pang, Jiangmiao and Lin, Dahua},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={913--922},
  year={2021}
}
```
