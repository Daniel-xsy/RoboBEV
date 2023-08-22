<img src="../figs/logo2.png" align="right" width="30%">

# RoboBEV Benchmark

## TPVFormer

| **Corruption** | **mIoU pts** | **mIoU vox** |
| :------------: | :-----: | :-----: |
| |
| Clean          | 26.84 | 52.06 |
| |
| Cam Crash      | 13.65 | 27.39 |
| Frame Lost     | 11.91 | 22.85 |
| Color Quant    | 21.49 | 38.16 |
| Motion Blur    | 20.79 | 38.64 |
| Brightness     | 25.76 | 49.00 |
| Low Light      | 19.78 | 37.38 |
| Fog            | 24.78 | 46.69 |
| Snow           | 9.49  | 19.39 |


## Experiment Log


### Camera Crash

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 18.76 | 37.49 |
| Moderate     | 9.19  | 19.88 |
| Hard         | 13.01 | 24.82 |
| Average      | 13.65 | 27.39 |


### Frame Lost

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 19.55 | 37.53 |
| Moderate     | 10.51 | 20.13 |
| Hard         | 5.69  | 10.89 |
| Average      | 11.91 | 22.85 |


### Color Quant

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 26.30 | 49.76 |
| Moderate     | 23.36 | 40.73 |
| Hard         | 14.83 | 23.99 |
| Average      | 21.49 | 38.16 |


### Motion Blur

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 25.70 | 48.04 |
| Moderate     | 20.03 | 36.77 |
| Hard         | 16.66 | 31.13 |
| Average      | 20.79 | 38.64 |


### Brightness

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 26.48 | 51.17 |
| Moderate     | 25.89 | 49.34 |
| Hard         | 24.91 | 46.51 |
| Average      | 25.76 | 49.00 |


### Low Light

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 23.44 | 44.20 |
| Moderate     | 20.39 | 38.31 |
| Hard         | 15.53 | 29.65 |
| Average      | 19.78 | 37.38 |


### Fog

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 25.30 | 48.04 |
| Moderate     | 24.92 | 47.10 |
| Hard         | 24.12 | 44.95 |
| Average      | 24.78 | 46.69 |


### Snow

| **Severity** | **mIoU pts** | **mIoU vox** |
| :----------: | :-----: |
| |
| Easy         | 14.50 | 28.52 |
| Moderate     | 7.34  | 15.51 |
| Hard         | 6.64  | 14.16 |
| Average      | 9.49  | 19.39 |



## References

```bib
@inproceedings{huang2023tri,
  title={Tri-perspective view for vision-based 3d semantic occupancy prediction},
  author={Huang, Yuanhui and Zheng, Wenzhao and Zhang, Yunpeng and Zhou, Jie and Lu, Jiwen},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={9223--9232},
  year={2023}
}
```
