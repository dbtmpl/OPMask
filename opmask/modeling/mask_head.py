import torch

from detectron2.layers import Conv2d, ConvTranspose2d

from detectron2.modeling import ROI_MASK_HEAD_REGISTRY
from detectron2.modeling.roi_heads.mask_head import mask_rcnn_inference
from detectron2.utils.events import get_event_storage
from fvcore.nn import weight_init as weight_init

from torch import nn
from torch.nn import functional as F

from ..utils.training_utils import mask_loss_ps, get_gt_masks
from ..layers.batch_norm import get_norm


class MetaCamMaskHead(nn.Module):
    def __init__(self, cfg, input_shape):
        """
        Meta class that allows to create different mask heads that use CAMs to steer mask predictions.

        :param cfg: Namespace containing all OPMask configs.
        :param input_shape: Provides information about the input feature map.
        """
        super().__init__()

        self.input_channels = input_shape.channels

        self.ps_mode = cfg.EXP.PS
        self.vis_period = cfg.VIS_PERIOD
        self.norm = cfg.MODEL.ROI_MASK_HEAD.NORM
        self.num_convs_down = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV
        self.conv_channels = cfg.MODEL.ROI_MASK_HEAD.CONV_DIM
        self.num_convs_up = cfg.MODEL.ROI_MASK_HEAD.NUM_CONV_UP

    def forward(self, x, cams, proposals):
        """
        Trivial forward function that takes FPN features, CAMs and Detectron2 Instances to perform
        mask prediction during training or inference time.
        """
        mask_logits = self.layers(x, cams)
        self.predictions_to_tensorboard(mask_logits, cams, proposals)
        return self.loss_or_inference(mask_logits, proposals)

    def layers(self, *inputs):
        """
        Everything that involves the actual forward pass of the mask head.
        """
        pass

    def loss_or_inference(self, mask_logits, proposals):
        """
        Gets predictions of the mask head and calculates the loss or performs inference depending
        on whether it is training or inference time.
        """
        if self.training:
            mask_losses = {}
            mask_losses.update(
                {f'mask_loss': mask_loss_ps(mask_logits, proposals, self.vis_period, self.ps_mode)}
            )
            return mask_losses
        else:
            mask_rcnn_inference(mask_logits, proposals)
            return proposals

    def get_down_convs(
            self,
            name,
            num_convs_down,
            input_channels,
            conv_channels,
            padding=1,
            stride=1,
            norm='',
            conv_type=Conv2d
    ):
        """
        Function to create and register a set of convolutional layers of arbitrary type. The
        default is the standard Conv2d.
        """
        layers = []
        for k in range(num_convs_down):
            conv = conv_type(
                input_channels if k == 0 else conv_channels,
                conv_channels,
                kernel_size=3,
                stride=stride,
                padding=padding,
                bias=not norm,
                norm=get_norm(norm, conv_channels),
                activation=F.relu
            )
            self.add_module(name + "{}".format(k + 1), conv)
            weight_init.c2_msra_fill(conv)
            layers.append(conv)
        return layers

    def get_conv_up(self, name, num_convs_up, conv_channels):
        """
        Function to create and register a set of ConvTranspose2d layers.
        """
        layers = []
        for k in range(num_convs_up):
            deconv = ConvTranspose2d(
                conv_channels,
                conv_channels,
                kernel_size=2,
                stride=2,
                padding=0,
            )
            self.add_module(name + "{}".format(k + 1), deconv)
            weight_init.c2_msra_fill(deconv)
            layers.append(deconv)
        return layers

    @staticmethod
    def get_predictor(input_channels, output_channels=1, init='normal'):
        """
        Creates a 1x1 convolution with weights either initialize with a `normal` distribution or
        with the `kaiming` initialization.
        """
        _predictor = Conv2d(input_channels, output_channels, kernel_size=1, stride=1, padding=0)
        if init == 'normal':
            nn.init.normal_(_predictor.weight, std=0.001)
            if _predictor.bias is not None:
                nn.init.constant_(_predictor.bias, 0)
        elif init == 'kaiming':
            weight_init.c2_msra_fill(_predictor)
        return _predictor

    def predictions_to_tensorboard(self, mask_logits, cams, proposals):
        """
        Sends CAMs and mask predictions to tensorboard while training.
        """
        if not self.training:
            return None
        storage = get_event_storage()
        if self.vis_period > 0 and storage.iter % self.vis_period == 0:
            mask_logits = F.interpolate(mask_logits, size=cams.size(2), mode='bilinear')
            gt_masks = get_gt_masks(proposals, cams.size(2), cams.device, self.training)
            vis_masks = torch.cat([cams, mask_logits.sigmoid(), gt_masks], dim=3)
            name = "Saved CAMs; Mask Logits; GT Mask"
            for idx, vis_mask in enumerate(vis_masks):
                vis_mask = torch.stack([vis_mask] * 3, axis=0)
                storage.put_image(name + f" ({idx})", vis_mask.squeeze())


@ROI_MASK_HEAD_REGISTRY.register()
class CamMaskHead(MetaCamMaskHead):

    def __init__(self, cfg, input_shape):
        """
        Mask Head of OPMask as discussed in the paper.
        """
        super().__init__(cfg, input_shape)

        self.mask_down = self.get_down_convs(
            name="mask_down",
            num_convs_down=self.num_convs_down,
            input_channels=self.input_channels,
            conv_channels=self.conv_channels,
            norm=self.norm
        )
        self.mask_up = self.get_conv_up(
            name="mask_up",
            num_convs_up=self.num_convs_up,
            conv_channels=self.conv_channels
        )
        self.predictor = self.get_predictor(
            input_channels=self.conv_channels
        )

    def layers(self, x, cam_raw):
        """
        Takes FPN features and CAMs to predict instance masks.
        """
        x = x + cam_raw

        for layer in self.mask_down:
            x = layer(x)

        for layer in self.mask_up:
            x = F.relu(layer(x))

        return self.predictor(x)
