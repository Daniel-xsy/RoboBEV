<img src="./figs/logo2.png" align="right" width="30%">

# Benchmark Results

## Corruption Error (CE)

### 📐 Metric Setting

The Corruption Error (CE) for a model $A$ under corruption $i$ across 3 severity levels is:

$$
\text{CE}_i^{\text{Model}A} = \frac{\sum^{3}_{l=1}((1 - \text{NDS})_{i,l}^{\text{Model}A})}{\sum^{3}_{l=1}((1 - \text{NDS})_{i,l}^{\text{Baseline}})} .
$$

The average CE for a model $A$ on all corruptions, i.e., mCE, is calculated as:

$$
\text{mCE} = \frac{1}{N}\sum^N_{i=1}\text{CE}_i ,
$$

where $N=8$ denotes the number of corruption types in our benchmark. We choose DETR3D as the baseline to calculate CE metric.

### 📊 Benchmark

| Model | NDS | mCE (%) $\downarrow$ | Cam Crash | Frame Lost | Color Quant | Motion Blur | Bright | Low Light | Fog | Snow |
| :- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [DETR3D](./results/DETR3D.md)<sup>:star:</sup> | 0.4224 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 | 100.00 |
| |
| [DETR3D<sub>CBGS</sub>](./results/DETR3D-w-cbgs.md) | 0.4341 | 99.21 | 98.15 | 98.90  | 99.15 | 101.62 | 97.47 | 100.28 | 98.23 | 99.85 |
| [BEVFormer<sub>Small</sub>](./results/BEVFormer-Small.md) | 0.4787 | 102.40 | 101.23 | 101.96  | 98.56 | 101.24 | 104.35 | 105.17 | 105.40 | 101.29 |
| [BEVFormer<sub>Base</sub>](./results/BEVFormer-Base.md) | 0.5174 | 97.97 | 95.87 | 94.42 | 95.13 | 99.54 | 96.97 | 103.76 | 97.42 | 100.69 |
| [PETR<sub>R50-p4</sub>](./results/PETR-r50.md) | 0.3665 | 111.01 | 107.55 | 105.92  | 110.33 | 104.93 | 119.36 | 116.84 | 117.02 | 106.13 |
| [PETR<sub>VoV-p4</sub>](./results/PETR-vov.md) | 0.4550 | 100.69 | 99.09 | 97.46  | 103.06 | 102.33 | 102.40 | 106.67 | 103.43 | 91.11 |
| [ORA3D](./results/ORA3D.md) | 0.4436 | 99.17 | 97.26 | 98.03 | 97.32 | 100.19 | 98.78 | 102.40 | 99.23 | 100.19 |
| [BEVDet<sub>R50</sub>](./results/BEVDet-r50.md) | 0.3770 | 115.12 | 105.22 | 109.19 | 111.27 | 108.18 | 123.96 | 123.34 | 123.83 | 115.93 |
| [BEVDet<sub>R101</sub>](./results/BEVDet-r101.md) | 0.3877 | 113.68 | 103.32 | 107.29 | 109.25 | 105.40 | 124.14 | 123.12 | 123.28 | 113.64 |
| [BEVDet<sub>R101-pt</sub>](./results/BEVDet-r101-FCOS3D-Pretrain.md) | 0.3780 | 112.80 | 105.84 | 108.68 | 101.99 | 100.97 | 123.39 | 119.31 | 130.21 | 112.04 |
| [BEVDet<sub>SwinT</sub>](./results/BEVDet-Swin-Tiny.md) | 0.4037 | 116.48 | 103.50 | 106.61 | 113.18 | 107.26 | 130.19 | 131.83 | 124.01 | 115.25 |
| [BEVDepth<sub>R50</sub>](./results/BEVDepth-r50.md) | 0.4058  | 110.02 | 103.09 | 106.26 | 106.24 | 102.02 | 118.72 |  114.26 | 116.57 | 112.98 |
| [BEVerse<sub>SwinT</sub>](./results/BEVerse-Tiny.md) | 0.4665 | 110.67 | 95.49 | 94.15 | 108.46 | 100.19 | 122.44 | 130.40 | 118.58 | 115.69 |
| [BEVerse<sub>SwinS</sub>](./results/BEVerse-Small.md) | 0.4951 | 107.82 | 92.93 | 101.61 | 105.42 | 100.40 | 110.14 | 123.12 | 117.46 | 111.48 |
| [PolarFormer<sub>R101</sub>](./results/PolarFormer-r101.md) | 0.4602 | 96.06 | 96.16 | 97.24 | 95.13 | 92.37 | 94.96 | 103.22 | 94.25 | 95.17 |
| [PolarFormer<sub>VoV</sub>](./results/PolarFormer-Vov.md)  | 0.4558 | 98.75 | 96.13 | 97.20 | 101.48 | 104.32 | 95.37 | 104.78 | 97.55 | 93.14 |
| [SRCN3D<sub>R101</sub>](./results/SRCN3D-r101.md) | 0.4286 | 99.67 | 98.77 | 98.96 | 97.93 | 100.71 | 98.80 | 102.72 | 99.54 | 99.91 |
| [SRCN3D<sub>VoV</sub>](./results/SRCN3D-Vov.md) | 0.4205 | 102.64 | 99.78 | 100.34 | 105.13 | 107.06 | 101.93 | 101.10 | 102.27 | 92.75	|
| [Sparse4D<sub>R101</sub>](./results/Sparse4D-r101.md) | 0.5438 | 100.01 | 99.80 | 99.91 | 98.05 | 102.00 | 100.30 | 103.83 | 100.46 | 95.72 |
| [SOLOFusion<sub>short</sub>](docs/results/SOLOFusion-short.md) | 0.3907 | 108.68 | 104.45 | 105.53 | 105.47 | 100.79 | 117.27 | 110.44 | 115.01 | 110.47 |
| [SOLOFusion<sub>long</sub>](docs/results/SOLOFusion-short.md) | 0.4850 | 97.99 | 95.80 | 101.54 | 93.83 | 89.11 | 100.00 | 99.61	| 98.70 | 105.35 |
| [SOLOFusion<sub>fusion</sub>](docs/results/SOLOFusion.md) | 0.5381 | 92.86 | 86.74 | 88.37 | 87.09 | 86.63 | 94.55 | 102.22 | 90.67	 | 106.64	 |
| |
| [FCOS3D<sub>finetune</sub>](docs/results/FCOS3D-ft.md) | 0.3949 | 107.82 | 100.14 | 101.69 | 108.84 | 101.24 | 113.07 | 118.27 | 109.71 | 109.61 |
| |
| [BEVFusion<sub>Cam</sub>](docs/results/BEVFusion-Camera.md) | 0.4122 | 109.02 | 101.15 | 104.72 | 106.07 | 98.27 | 118.34 | 123.70 | 114.31 | 105.59 |
| [BEVFusion<sub>LiDAR</sub>](docs/results/BEVFusion-LiDAR.md) | 0.6928 | - | - | - | - | - | - | - | - | - |
| [BEVFusion<sub>C+L</sub>](docs/results/BEVFusion-Fusion.md) | 0.7138 | 43.80 | 42.53 | 41.50 | 43.32 | 41.19 | 49.72 | 44.54 | - | - |

## Resilience Rate (RR)

### 📐Metric Setting
The Resilience Rate (RR) for a model $A$ under corruption $i$ across 3 severity levels is:

$$
\text{RR}_i^{\text{Model}A} = \frac{\sum^{3}_{l=1}(\text{NDS}_{i,l}^{\text{Model}A})}{3\times \text{NDS}_{\text{clean}}^{\text{Model}A}} .

$$
The average RR for a model $A$ on all corruptions, i.e., mRR, is calculated as:

$$
\text{mRR} = \frac{1}{N}\sum^N_{i=1}\text{RR}_i ,
$$

where $N=8$ denotes the number of corruption types in our benchmark.

### 📊 Benchmark

| Model | NDS |mRR (%) $\uparrow$ | Cam Crash | Frame Lost | Color Quant | Motion Blur | Bright | Low Light | Fog | Snow |
| :- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| [DETR3D](./results/DETR3D.md)<sup>:star:</sup> | 0.4224 | 70.77 | 67.68 | 61.65 | 75.21 | 63.00 | 94.74 | 65.96 | 92.61 | 45.29 |
| |
| [DETR3D<sub>CBGS</sub>](./results/DETR3D-w-cbgs.md) | 0.4341 | 70.02 | 68.90 | 61.85  | 74.52 | 58.56 | 95.69 | 63.72 | 92.61 | 44.34 |
| [BEVFormer<sub>Small</sub>](./results/BEVFormer-Small.md) | 0.4787 | 59.07 | 57.89 | 51.37 | 68.41  | 53.69 | 78.15 | 50.41 | 74.85 | 37.79 |
| [BEVFormer<sub>Base</sub>](./results/BEVFormer-Base.md) | 0.5174 | 60.40 | 60.96 | 58.31 | 67.82 | 52.09 | 80.87 | 48.61 | 78.64 | 35.89 |
| [PETR<sub>R50-p4</sub>](./results/PETR-r50.md) | 0.3665 | 61.26 | 63.30 | 59.10  | 67.45 | 62.73 | 77.52 | 42.86 | 78.47 | 38.66 |
| [PETR<sub>VoV-p4</sub>](./results/PETR-vov.md) | 0.4550 | 65.03 | 64.26 | 61.36  | 65.23 | 54.73 | 84.79 | 50.66 | 81.38 | 57.85 |
| [ORA3D](./results/ORA3D.md) | 0.4436 | 68.63 | 68.87 | 61.99 | 75.74 | 59.67 | 91.86 | 58.90 | 89.25 | 42.79 |
| [BEVDet<sub>R50</sub>](./results/BEVDet-r50.md) | 0.3770 | 51.83 | 65.94 | 51.03 | 63.87 | 54.67 | 68.04 | 29.23 | 65.28 | 16.58 |
| [BEVDet<sub>R101</sub>](./results/BEVDet-r101.md) | 0.3877 | 53.12 | 67.63 | 53.26 | 65.67 | 58.42 | 65.88 | 28.84 | 64.35 | 20.89 |
| [BEVDet<sub>R101-pt</sub>](./results/BEVDet-r101-FCOS3D-Pretrain.md) | 0.3780 | 56.35 | 64.60 | 51.90 | 80.45 | 68.52 | 68.76 | 36.85 | 54.84 | 24.84 |
| [BEVDet<sub>SwinT</sub>](./results/BEVDet-Swin-Tiny.md) | 0.4037 | 46.26 | 64.63 | 52.39 | 56.43 | 52.71 | 54.27 | 12.14 | 60.69 | 16.84 |
| [BEVDepth<sub>R50</sub>](./results/BEVDepth-r50.md) | 0.4058  | 56.82 | 65.01 | 52.76 | 67.79 | 61.93 | 70.95 |  43.30 | 71.54 | 21.27 |
| [BEVerse<sub>SwinT</sub>](./results/BEVerse-Tiny.md) | 0.4665 | 48.60 | 68.19 | 65.10 | 55.73 | 56.74 | 56.93 | 12.71 | 59.61 | 13.80 |
| [BEVerse<sub>SwinS</sub>](./results/BEVerse-Small.md) | 0.4951 | 49.57 | 67.95 | 50.19 | 56.70 | 53.16 | 68.55 | 22.58 | 57.54 | 19.89 |
| [PolarFormer<sub>R101</sub>](./results/PolarFormer-r101.md) | 0.4602 | 70.88 | 68.08 | 61.02 | 76.25 | 69.99 | 93.52 | 55.50 | 92.61 | 50.07 |
| [PolarFormer<sub>VoV</sub>](./results/PolarFormer-Vov.md)  | 0.4558 | 67.51 | 68.78 | 61.67 | 67.49 | 51.43 | 93.90 | 53.55 | 89.10 | 54.15 |
| [SRCN3D<sub>R101</sub>](./results/SRCN3D-r101.md) | 0.4286 | 70.23 | 68.76 | 62.55 | 77.41 | 60.87 | 95.05 | 60.43 | 91.93 | 44.80 |
| [SRCN3D<sub>VoV</sub>](./results/SRCN3D-Vov.md) | 0.4205 | 67.95 | 68.37 | 61.33 | 67.23 | 50.96 | 92.41 | 54.08 | 89.75 | 59.43	|
| [Sparse4D<sub>R101</sub>](./results/Sparse4D-r101.md) | 0.5438 | 55.04 | 52.83 | 48.01 | 60.87 | 46.23 | 73.26 | 46.16 | 71.42 | 41.54 |
| [SOLOFusion<sub>short</sub>](docs/results/SOLOFusion-short.md) | 0.3907 | 61.45 | 65.04 | 56.18 | 71.77 | 66.62 | 75.92 | 52.03 | 76.73 | 27.28 |
| [SOLOFusion<sub>long</sub>](docs/results/SOLOFusion-short.md) | 0.4850 | 64.42 | 65.13| 51.34 | 74.19 | 71.34 | 82.52 | 58.02	| 82.29 | 30.52 |
| [SOLOFusion<sub>fusion</sub>](docs/results/SOLOFusion.md) | 0.5381 | 64.53 | 70.73 | 64.37 | 75.41 | 67.68 | 80.45 | 48.80 | 83.26 | 25.57 |
| |
| [FCOS3D<sub>finetune</sub>](docs/results/FCOS3D-ft.md) | 0.3949 | 62.09 | 72.14 | 62.78 | 65.18 | 65.08 | 81.49 | 37.17 | 84.10 | 28.77 |
| |
| [BEVFusion<sub>Cam</sub>](docs/results/BEVFusion-Camera.md) | 0.4122 | 57.81 | 67.37 | 54.71 | 67.03 | 67.64 | 70.40 | 26.10 | 73.77 | 35.44 |
| [BEVFusion<sub>LiDAR</sub>](docs/results/BEVFusion-LiDAR.md) | 0.6928 | - | - | - | - | - | - | - | - | - |
| [BEVFusion<sub>C+L</sub>](docs/results/BEVFusion-Fusion.md) | 0.7138 | 97.41 | 97.55 | 97.10 | 98.68 | 97.74 | 98.32 | 95.08 | - | - |
