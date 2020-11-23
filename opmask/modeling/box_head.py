import numpy as np

import torch.nn as nn
from torch.nn import functional as F

import fvcore.nn.weight_init as weight_init

from detectron2.modeling.roi_heads.box_head import ROI_BOX_HEAD_REGISTRY
from detectron2.layers import Linear, Conv2d

from ..layers.batch_norm import get_norm


@ROI_BOX_HEAD_REGISTRY.register()
class CAMBoxHeadConv(nn.Module):
    """
    Copyright (c) Facebook, Inc. and its affiliates.
    Adapted Detectron2 class.

    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu) that allows calculating class activation maps (CAMs).
    """

    def __init__(self, cfg, input_shape, num_classes, cls_agnostic_bbox_reg, box_dim=4):
        """
        The following attributes are parsed from config:
            num_conv, num_fc: the number of conv/fc layers
            conv_dim/fc_dim: the dimension of the conv/fc layers
            norm: normalization for the conv layers
        """
        super().__init__()

        # fmt: off
        num_conv = cfg.MODEL.ROI_BOX_HEAD.NUM_CONV
        conv_dim = cfg.MODEL.ROI_BOX_HEAD.CONV_DIM
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        norm = cfg.MODEL.ROI_BOX_HEAD.NORM
        # fmt: on
        assert num_conv + num_fc > 0

        num_bbox_reg_classes = 1 if cls_agnostic_bbox_reg else num_classes
        self.num_classes = num_classes
        self.pred_reg = num_bbox_reg_classes * box_dim

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)

        self.conv_norm_relus = []
        for k in range(num_conv):
            dim_in = conv_dim if self._output_size[0] * 2 ** k >= conv_dim else self._output_size[0] * 2 ** k
            dim_out = conv_dim if self._output_size[0] * 2 ** (k + 1) >= conv_dim else self._output_size[0] * 2 ** (
                    k + 1)
            conv = Conv2d(
                dim_in,
                dim_out,
                kernel_size=3,
                padding=1,
                bias=not norm,
                norm=get_norm(norm, dim_out),
                activation=F.relu,
            )
            self.add_module("conv{}".format(k + 1), conv)
            self.conv_norm_relus.append(conv)
        self._output_size = (conv_dim, self._output_size[1], self._output_size[2])

        self.fcs = []
        for k in range(num_fc):
            fc = Linear(np.prod(self._output_size), fc_dim)
            self.add_module("fc{}".format(k + 1), fc)
            self.fcs.append(fc)
            self._output_size = fc_dim

        self.input_size = self.output_size
        if not isinstance(self.input_size, int):
            self.input_size = self.input_size[0]

        self.cls_score, self.bbox_pred = self.init_pred_layers()

        for layer in self.conv_norm_relus:
            weight_init.c2_msra_fill(layer)
        for layer in self.fcs:
            weight_init.c2_xavier_fill(layer)

        nn.init.normal_(self.cls_score.weight, std=0.01)
        nn.init.normal_(self.bbox_pred.weight, std=0.001)
        for layer in [self.cls_score, self.bbox_pred]:
            nn.init.constant_(layer.bias, 0)

    def init_pred_layers(self):
        cls_score = Linear(self.input_size, self.num_classes + 1)
        bbox_pred = Linear(self.input_size, self.pred_reg)
        return cls_score, bbox_pred

    def forward(self, x):
        bs = x.size(0)
        for layer in self.conv_norm_relus:
            x = layer(x)

        x_pool = x.mean(dim=[2, 3])  # pooled to 1x1

        return self.cls_score(x_pool).view(bs, self.num_classes + 1), self.bbox_pred(x_pool).view(bs, self.pred_reg), x

    @property
    def output_size(self):
        return self._output_size


def build_box_head(cfg, input_shape, num_classes, cls_agnostic_bbox_reg):
    """
    Build a box head defined by `cfg.MODEL.ROI_BOX_HEAD.NAME`.
    """
    name = cfg.MODEL.ROI_BOX_HEAD.NAME
    return ROI_BOX_HEAD_REGISTRY.get(name)(cfg, input_shape, num_classes, cls_agnostic_bbox_reg)
