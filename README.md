# Robust BEV Object Detection

## Getting Started
Coming soon.

## Evaluate under corruption

To evaluate model under corruptions, add `corruptions` in mmdet config file, and run the folowwing command:
```shell
bash tools/dist_test.sh projects/configs/bevformer/bevformer_base.py path/to/model 4
```
Results will be saved in `./log` folder with the prefix of model name.
