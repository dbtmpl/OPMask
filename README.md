# OPMask
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=python)](https://www.python.org/) [![NumPy 1.18.5](https://img.shields.io/badge/NumPy-1.18.5-blue)](https://numpy.org/doc/1.18/)
 [![Torch 1.7.0](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/) [![Detectron2](https://img.shields.io/badge/Detectron2-v0.1-orange)](https://pytorch.org/)

This repository provides the official implementation of the paper:
> **[Prior to Segment: Foreground Cues for Novel Objects in Partially Supervised Instance Segmentation](#)**<br>
> *[David Biertimpel](https://scholar.google.com/citations?user=AIu7ihgAAAAJ&hl=en), †[Sindi Shkodrani](https://scholar.google.nl/citations?user=fFVkKNgAAAAJ&hl=en), *[Anil Baslamisli](https://scholar.google.nl/citations?user=mc4l2J4AAAAJ&hl=en) and †[Nóra Baka](https://scholar.google.com/citations?user=ahfzQHEAAAAJ&hl=en) <br>
> *University of Amsterdam, †TomTom<br>
> pre-print : coming soon <br>

![Image](assets/images/architecture.png?raw=true)

## Abstract
Instance segmentation methods require large datasets with expensive instance-level mask labels. This makes partially supervised learning appealing in settings where abundant box and limited mask labels are available. To improve mask predictions with limited labels, we modify a Mask R-CNN by introducing an object mask prior (OMP) for the mask head. We show that a conventional class-agnostic mask head has difficulties learning foreground for classes with box-supervision only. Our OMP resolves this by providing the mask head with the general concept of foreground implicitly learned by the box classification head under the supervision of all classes. This helps the class-agnostic mask head to focus on the primary object in a region of interest (RoI) and improves generalization to novel classes. We test our approach on the COCO dataset using different splits of strongly and weakly supervised classes. Our approach significantly improves over the Mask R-CNN baseline and obtains competitive performance with the state-of-the-art, while offering a much simpler architecture. 

## Setup
OPMask relies on [PyTorch](https://pytorch.org/) and [Detectron2](https://github.com/facebookresearch/detectron2). Detectron2 requires [OpenCV](https://opencv.org/) that may need to be installed manually. 
Currently, our implementation only supports the Detectron2 version with which we conducted our research. We intend to support the latest version as soon as possible. 
For training on [COCO](https://cocodataset.org/) installing `pycocotools` is required.
After setting up an environment in your preferred way and installing PyTorch, the correct version of Detectron2 can be installed as follows:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@e49c555a046a7495db58d327f34058e7dc858275'
```

## Training
Training OPMask is simple. In `configs/` we provide different configuration files for reproducing the models in the paper. Training can be executed as follows:
```Train model
python start.py --config-file=configs/opmask_R50_FPN_130k.yaml opts MODEL.DEVICE 'cpu'
```
With `opts` configs in `configs/` or default Detectron2 configs can be overwritten. 
For each model a folder `output/OPMask/{dataset}_{exp-id}` is created. With the flag `--exp-id` the folder name can be personalized. The datasets must be configured with the instructions in `datasets/`. 

![Image](assets/images/qualitative_overlay.png?raw=true)

## Evaluation
By default each model is evaluated every `TEST.EVAL_PERIOD` iterations and after training. To evaluate a trained model the flag `--eval-only` can be used. Note: the configs (e.g. `exp-id`) need to point to the folder of the trained model.

## <a name="Citing OPMask"></a> Citation
For citing our paper please use the following BibTeX entry:
```
Coming Soon
```

## Acknowledgements
Special thanks to the AddamsFamily and the TomTom MAPS-Autonomous Driving Team. Thanks to TomTom for providing exhaustive computational resources.
