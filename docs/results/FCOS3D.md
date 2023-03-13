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


## FCOS3D

| **Corruption** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :------------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Clean          | 0.3773 | 0.2979 | 0.7899 | 0.2606 | 0.4988 | 1.2869 | 0.1671 |
| |
| Cam Crash      | 0.2684    | 0.1080    | 0.8167     | 0.2929     | 0.5372     | 1.2684     | 0.2092     |
| Frame Lost     | 0.2390    | 0.0849    | 0.8283     | 0.3532     | 0.5511     | 1.3366     | 0.3014     |
| Color Quant    | 0.2475    | 0.1438    | 0.9257     | 0.3627     | 0.6409     | 1.3901     | 0.3256     |
| Motion Blur    | 0.2470    | 0.1396    | 0.8616     | 0.3535     | 0.6853     | 1.3045     | 0.3278     |
| Brightness     | 0.3155    | 0.2136    | 0.8406     | 0.2803     | 0.6101     | 1.4281     | 0.1821     |
| Low Light      | 0.1647    | 0.0582    | 0.8855     | 0.4768     | 0.7636     | 1.3272     | 0.5178     |
| Fog            | 0.3203    | 0.2141    | 0.8462     | 0.2724     | 0.5584     | 1.2880     | 0.1905     |
| Snow           | 0.1400    | 0.0495    | 0.9715     | 0.5307     | 0.7917     | 1.2423     | 0.5535     |


## Experiment Log

> Time: Sat Mar 11 16:33:22 2023


### Camera Crash

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3101    | 0.1691    | 0.7978     | 0.2656     | 0.5159     | 1.1733     | 0.1646     |
| Moderate     | 0.2507    | 0.0752    | 0.8498     | 0.2676     | 0.5646     | 1.2776     | 0.1868     |
| Hard         | 0.2444    | 0.0798    | 0.8024     | 0.3454     | 0.5310     | 1.3544     | 0.2763     |
| Average      | 0.2684    | 0.1080    | 0.8167     | 0.2929     | 0.5372     | 1.2684     | 0.2092     |


### Frame Lost

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3182    | 0.1836    | 0.8036     | 0.2619     | 0.5051     | 1.2885     | 0.1654     |
| Moderate     | 0.2375    | 0.0607    | 0.8227     | 0.3233     | 0.5301     | 1.4097     | 0.2524     |
| Hard         | 0.1614    | 0.0103    | 0.8586     | 0.4743     | 0.6182     | 1.3117     | 0.4863     |
| Average      | 0.2390    | 0.0849    | 0.8283     | 0.3532     | 0.5511     | 1.3366     | 0.3014     |


### Color Quant

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3480    | 0.2563    | 0.8223     | 0.2694     | 0.5267     | 1.3456     | 0.1832     |
| Moderate     | 0.2428    | 0.1345    | 0.9210     | 0.3663     | 0.6283     | 1.3698     | 0.3290     |
| Hard         | 0.1518    | 0.0406    | 1.0338     | 0.4523     | 0.7677     | 1.4548     | 0.4647     |
| Average      | 0.2475    | 0.1438    | 0.9257     | 0.3627     | 0.6409     | 1.3901     | 0.3256     |


### Motion Blur

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3419    | 0.2498    | 0.8006     | 0.2681     | 0.5784     | 1.2686     | 0.1836     |
| Moderate     | 0.2186    | 0.1052    | 0.8879     | 0.3702     | 0.7360     | 1.2590     | 0.3456     |
| Hard         | 0.1805    | 0.0637    | 0.8962     | 0.4222     | 0.7415     | 1.3859     | 0.4542     |
| Average      | 0.2470    | 0.1396    | 0.8616     | 0.3535     | 0.6853     | 1.3045     | 0.3278     |


### Brightness

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3531    | 0.2606    | 0.8012     | 0.2627     | 0.5496     | 1.3825     | 0.1584     |
| Moderate     | 0.3128    | 0.2102    | 0.8384     | 0.2809     | 0.6123     | 1.4462     | 0.1916     |
| Hard         | 0.2806    | 0.1701    | 0.8823     | 0.2974     | 0.6685     | 1.4555     | 0.1963     |
| Average      | 0.3155    | 0.2136    | 0.8406     | 0.2803     | 0.6101     | 1.4281     | 0.1821     |


### Low Light

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2096    | 0.0944    | 0.8605     | 0.4198     | 0.6561     | 1.4420     | 0.4392     |
| Moderate     | 0.1773    | 0.0565    | 0.8818     | 0.4303     | 0.7299     | 1.4240     | 0.4680     |
| Hard         | 0.1073    | 0.0237    | 0.9142     | 0.5803     | 0.9047     | 1.1155     | 0.6463     |
| Average      | 0.1647    | 0.0582    | 0.8855     | 0.4768     | 0.7636     | 1.3272     | 0.5178     |


### Fog

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.3357    | 0.2382    | 0.8308     | 0.2674     | 0.5492     | 1.2965     | 0.1862     |
| Moderate     | 0.3220    | 0.2155    | 0.8442     | 0.2730     | 0.5509     | 1.2867     | 0.1899     |
| Hard         | 0.3033    | 0.1888    | 0.8636     | 0.2768     | 0.5750     | 1.2809     | 0.1953     |
| Average      | 0.3203    | 0.2141    | 0.8462     | 0.2724     | 0.5584     | 1.2880     | 0.1905     |


### Snow

| **Severity** | **NDS** | **mAP** | **mATE** | **mASE** | **mAOE** | **mAVE** | **mAAE** |
| :----------: | :-----: | :-----: | :------: | :------: | :------: | :------: | :------: |
| |
| Easy         | 0.2065    | 0.0890    | 0.9338     | 0.3883     | 0.6816     | 1.4209     | 0.3762     |
| Moderate     | 0.0923    | 0.0274    | 0.9983     | 0.6650     | 0.8237     | 1.1056     | 0.7272     |
| Hard         | 0.1213    | 0.0322    | 0.9825     | 0.5389     | 0.8699     | 1.2004     | 0.5571     |
| Average      | 0.1400    | 0.0495    | 0.9715     | 0.5307     | 0.7917     | 1.2423     | 0.5535     |



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
