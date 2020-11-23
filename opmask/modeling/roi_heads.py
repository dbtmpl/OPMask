from typing import Dict, List, Union, Optional, Tuple

import torch
from torch.nn import functional as F

from detectron2.layers import ShapeSpec
from detectron2.structures import Instances, ImageList, Boxes

from detectron2.modeling import ROI_HEADS_REGISTRY, StandardROIHeads
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads import select_foreground_proposals
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputs

from .box_head import build_box_head
from ..utils.cam_utils import process_cams_batch


@ROI_HEADS_REGISTRY.register()
class FPNCamRoiHeads(StandardROIHeads):
    """
    Copyright (c) Facebook, Inc. and its affiliates.
    Adapted Detectron2 class.

    Small adjustments to the StandardROIHeads to be able to calculate and process
    class activation maps (CAMs)
    """

    def __init__(self, cfg, input_shape):
        super().__init__(cfg, input_shape)
        self.cam_res = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION

    def _init_box_head(self, cfg, input_shape):
        # fmt: off
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales = tuple(1.0 / input_shape[k].stride for k in self.in_features)
        sampling_ratio = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        self.train_on_pred_boxes = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in self.in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        self.box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )

        self.box_head = build_box_head(
            cfg,
            ShapeSpec(channels=in_channels, height=pooler_resolution, width=pooler_resolution),
            self.num_classes,
            self.cls_agnostic_bbox_reg,
        )

    def forward(
            self,
            images: ImageList,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances],
            targets: Optional[List[Instances]] = None,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training:
            assert targets
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            losses, box_features = self._forward_box(features, proposals)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            losses.update(self._forward_mask(features, box_features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances, box_features = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.

            pred_instances = self.forward_with_given_boxes(features, box_features, pred_instances)
            return pred_instances, {}

    def _forward_box(
            self,
            features: Dict[str, torch.Tensor],
            proposals: List[Instances]
    ) -> Union[Tuple[Dict[str, torch.Tensor], torch.Tensor], List[Instances]]:

        features = [features[f] for f in self.in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        pred_class_logits, pred_proposal_deltas, box_features = self.box_head(box_features)

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )

        if self.training:
            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = outputs.predict_boxes_for_gt_classes()
                    for proposals_per_image, pred_boxes_per_image in zip(proposals, pred_boxes):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return outputs.losses(), box_features
        else:
            pred_instances, indices = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh, self.test_detections_per_img
            )
            return pred_instances, box_features[indices]

    def _forward_mask(
            self,
            features: Dict[str, torch.Tensor],
            box_features: torch.Tensor,
            instances: List[Instances],
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        if not self.mask_on:
            return {} if self.training else instances

        features = [features[f] for f in self.in_features]

        if self.training:
            # The loss is only defined on positive proposals.
            proposals, fg_selection_masks = select_foreground_proposals(instances, self.num_classes)
            proposal_boxes = [x.proposal_boxes for x in proposals]
            mask_features = self.mask_pooler(features, proposal_boxes)
            box_features = box_features[torch.cat(fg_selection_masks, dim=0)]
            cam_raw_batch = self.calculate_cams(mask_features, box_features, proposals, self.training)

            return self.mask_head(mask_features, cam_raw_batch, proposals)
        else:
            pred_boxes = [x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(features, pred_boxes)
            cam_raw_batch = self.calculate_cams(mask_features, box_features, instances, self.training)

            return self.mask_head(mask_features, cam_raw_batch, instances)

    def calculate_cams(self, mask_features, box_features, instances, is_train):
        if mask_features.size(0) == 0:
            cams_batch = torch.zeros(0, 1, self.cam_res, self.cam_res, device=mask_features.device)
        else:
            w_cls = self.box_head.cls_score.weight.clone().detach()
            cams_batch = F.conv2d(box_features, weight=w_cls[..., None, None])
            cams_batch = process_cams_batch(cams_batch, instances, self.cam_res, is_train=is_train)
        return cams_batch

    def forward_with_given_boxes(
            self,
            features: Dict[str, torch.Tensor],
            box_features: torch.Tensor,
            instances: List[Instances]
    ) -> List[Instances]:
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, box_features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances
