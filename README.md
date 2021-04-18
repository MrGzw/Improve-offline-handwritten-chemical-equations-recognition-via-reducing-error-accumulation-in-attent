## 介绍
当前基于图像的序列识别的主流模型是耦合注意网络，已被证明是非常成功的。但是，传统的耦合注意机制通常会遭受严重的错误累积问题，这不仅是因为attention map的生成过程，而且还由于解码器单元的输入与历史解码器信息有关。 为了减少这种现象，本模型通过结合解耦注意网络和计划采样，通过减少因依赖历史解码器信息而引起的错误积累和传播，进一步提高了离线手写化学方程式识别的准确率。  

## Introduction
Current mainstream model to deal with image-based sequence recognition is the coupled attention network, which has been proved to have remarkable success.But, traditional coupled attention mechanism usually suffers from serious error accumulation problem, because not only the generation of attention map but also the input of decoder unit is related to the historical decoder information. To reduce this phenomenon, some technologies like decoupled attention network and scheduled sampling have been proposed to deal with the generation of error accumulation in different periods respectively. In this thesis, an advanced decoupled attention network and a training mechanism scheduled sampling are combined to further improve the accuracy rate in offline handwritten chemical equation recognition by reducing the error accumulation and propagation caused by its dependence of historical decoder information.   


## Training and Testing  
Modify the path in configuration files (`cfgs_hw.py` for handwritten). Make sure the import is correct in `line 12, main.py`. Then:  
run main.py

## Requirements

better to use [Anaconda](https://www.anaconda.com/) to manage your libraries.

- [Python 3.7](https://www.python.org/) (The data augmentation toolkit does not support python3)
- [PyTorch](https://pytorch.org/) (We have tested 0.4.1 and 1.1.0)
- [TorchVision](https://pypi.org/project/torchvision/)
- [OpenCV](https://opencv.org/)
- [PIL (Pillow)](https://pillow.readthedocs.io/en/stable/#)
- [Colour](https://pypi.org/project/colour/)
- [LMDB](https://pypi.org/project/lmdb/)
- [editdistance]
