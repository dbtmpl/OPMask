# OPMask
[![Python 3.8](https://img.shields.io/badge/Python-3.8-3776AB.svg?logo=python)](https://www.python.org/) [![NumPy 1.18.5](https://img.shields.io/badge/NumPy-1.19.4-blue)](https://numpy.org/doc/1.18/)
 [![Torch 1.7.0](https://img.shields.io/badge/PyTorch-1.7.0-orange)](https://pytorch.org/) [![Detectron2](https://img.shields.io/badge/Detectron2-v0.1-orange)](https://pytorch.org/)

This repository provides the official implementation of the paper:
> **[Prior to Segment: Foreground Cues for Weakly Annotated Classes in Partially Supervised Instance Segmentation](https://arxiv.org/abs/2011.11787)** <br>
> *†[David Biertimpel](https://scholar.google.com/citations?user=AIu7ihgAAAAJ&hl=en), †[Sindi Shkodrani](https://scholar.google.nl/citations?user=fFVkKNgAAAAJ&hl=en), *[Anil S. Baslamisli](https://scholar.google.nl/citations?user=mc4l2J4AAAAJ&hl=en) and †[Nóra Baka](https://scholar.google.com/citations?user=ahfzQHEAAAAJ&hl=en) <br>
> *University of Amsterdam, †TomTom<br>
> pre-print : https://arxiv.org/abs/2011.11787 <br>

![Image](assets/images/architecture.png?raw=true)

## Abstract
Instance segmentation methods require large datasets with expensive and thus limited instance-level mask labels. Partially supervised instance segmentation aims to improve mask prediction with limited mask labels by utilizing the more abundant weak box labels. In this work, we show that a class agnostic mask head, commonly used in partially supervised instance segmentation, has difficulties learning a general concept of foreground for the weakly annotated classes using box supervision only. To resolve this problem we introduce an object mask prior (OMP) that provides the mask head with the general concept of foreground implicitly learned by the box classification head under the supervision of all classes. This helps the class agnostic mask head to focus on the primary object in a region of interest (RoI) and improves generalization to the weakly annotated classes. We test our approach on the COCO dataset using different splits of strongly and weakly supervised classes. Our approach significantly improves over the Mask R-CNN baseline and obtains competitive performance with the state-of-the-art, while offering a much simpler architecture.

## Setup
OPMask relies on [PyTorch](https://pytorch.org/) and [Detectron2](https://github.com/facebookresearch/detectron2). 
Currently, our implementation only supports the Detectron2 version with which we conducted our research. We intend to support the latest version as soon as possible. 
For training on [COCO](https://cocodataset.org/) installing `pycocotools` is required.
After setting up a `Python 3.8` environment in your preferred way, all dependencies can be installed using the `requirements.txt`:
```
pip install -r requirements.txt
```
The correct version of Detectron2 can also be installed manually as follows:
```
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git@e49c555a046a7495db58d327f34058e7dc858275'
```

## Training
Training OPMask is simple. In `configs/` we provide different configuration files for reproducing the models in the paper. To test if your installation is correct, you can start a test training on CPU:
```Train model
python start.py --config-file=configs/opmask_R50_FPN_130k.yaml --exp-id=test_run MODEL.DEVICE 'cpu' SOLVER.IMS_PER_BATCH 1
```
Configs can be overwritten by appending `key value` pairs to the command (e.g. `MODEL.DEVICE 'cpu'` or `TEST.EVAL_PERIOD 5000`).
For each run a folder `output/OPMask/{dataset}_{exp-id}` is created. With the `--exp-id` flag the folder name can be personalized. The datasets must be configured with the instructions in `datasets/`. 

![Image](assets/images/qualitative_overlay_voc.png?raw=true)

## Evaluation
By default each model is evaluated every `TEST.EVAL_PERIOD` iterations and after training. To evaluate a trained model the `--eval-only` flag can be used. Note: the configs (e.g. `exp-id`) need to point to the folder of the trained model.

## <a name="Citing OPMask"></a> Citation
For citing our paper please use the following BibTeX entry:
```
@misc{biertimpel2021prior,
      title={Prior to Segment: Foreground Cues for Weakly Annotated Classes in Partially Supervised Instance Segmentation}, 
      author={David Biertimpel and Sindi Shkodrani and Anil S. Baslamisli and Nóra Baka},
      year={2021},
      eprint={2011.11787},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Acknowledgements
Special thanks to the ADdamsFamily and the TomTom MAPS-Autonomous Driving Team. Thanks to TomTom for providing exhaustive computational resources.
