# GraphPose-Fi
## Introduction
This repository contains the implementation of the paper: Graph-based 3D Human Pose Estimation using WiFi Signals.
## Requirements
The code is devoloped and tested under the following environemnt:
- Python 3.10.16
- PyTorch 2.7.0
- CUDA 12.8
## Quick Start
### Dataset
We use the MM-Fi dataset. Please request access and follow the official instructions: [MM-Fi](https://ntu-aiot-lab.github.io/mm-fi). Place the downloaded data under your dataset root and set the path in the config:
```shell
dataset_root:
```
### Train
Simply train the algorithm using the following command:
```shell
$ python train.py --config_file config/mmfi/pose_config_p1s1.yaml
```
You can pass training parameters via argparse flags on the command line. Use the exact flag names defined in train.py (e.g., --learning_rate, --batch_size):
```shell
$ python train.py --config_file config/mmfi/pose_config_p1s1.yaml --learning_rate 3e-4 --batch_size 256
```
## Acknowledgements
The implementation of this repo is based on [DT-Pose](https://github.com/cseeyangchen/DT-Pose) and [GraFormer](https://github.com/Graformer/GraFormer). Thanks them for releasing the codes.

