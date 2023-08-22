<img src="../figs/logo2.png" align="right" width="30%">

# RoboBEV Benchmark

## SurroundOcc


| **Corruption** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Clean          | 0.3149 | 0.2030 |
| |
| Cam Crash      | 0.1996 | 0.1160 |
| Frame Lost     | 0.1810 | 0.1000 |
| Color Quant    | 0.2584 | 0.1403 |
| Motion Blur    | 0.2257 | 0.1241 |
| Brightness     | 0.3072 | 0.1918 |
| Low Light      | 0.2479 | 0.1215 |
| Fog            | 0.2964 | 0.1842 |
| Snow           | 0.1834 | 0.0739 |


## Experiment Log


### Camera Crash

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.2358 | 0.1421 |
| Moderate     | 0.1747 | 0.1015 |
| Hard         | 0.1884 | 0.1044 |
| Average      | 0.1996 | 0.1160 |


### Frame Lost

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.2522 | 0.1549 |
| Moderate     | 0.1686 | 0.0907 |
| Hard         | 0.1223 | 0.0545 |
| Average      | 0.1810 | 0.1000 |


### Color Quant

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.3057 | 0.1906 |
| Moderate     | 0.2679 | 0.1491 |
| Hard         | 0.2016 | 0.0814 |
| Average      | 0.2584 | 0.1403 |


### Motion Blur

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.2854 | 0.1797 |
| Moderate     | 0.2107 | 0.1093 |
| Hard         | 0.1810 | 0.0834 |
| Average      | 0.2257 | 0.1241 |


### Brightness

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.3133 | 0.2008 |
| Moderate     | 0.3081 | 0.1925 |
| Hard         | 0.3004 | 0.1822 |
| Average      | 0.3072 | 0.1918 |


### Low Light

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.2734 | 0.1537 |
| Moderate     | 0.2505 | 0.1249 |
| Hard         | 0.2199 | 0.0861 |
| Average      | 0.2479 | 0.1215 |


### Fog

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.3047 | 0.1909 |
| Moderate     | 0.2988 | 0.1860 |
| Hard         | 0.2858 | 0.1759 |
| Average      | 0.2964 | 0.1842 |


### Snow

| **Severity** | **IoU** | **mIoU** |
| :------------: | :-----: | :-----: |
| |
| Easy         | 0.2082 | 0.1007 |
| Moderate     | 0.1770 | 0.0660 |
| Hard         | 0.1650 | 0.0551 |
| Average      | 0.1834 | 0.0739 |



## References

```bib
@article{wei2023surroundocc, 
      title={SurroundOcc: Multi-Camera 3D Occupancy Prediction for Autonomous Driving}, 
      author={Yi Wei and Linqing Zhao and Wenzhao Zheng and Zheng Zhu and Jie Zhou and Jiwen Lu},
      journal={arXiv preprint arXiv:2303.09551},
      year={2023}
}
```
