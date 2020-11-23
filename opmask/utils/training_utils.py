import torch
from torch.nn import functional as F

from detectron2.layers import cat
from detectron2.utils.events import get_event_storage

from ..utils.class_splits import VOC_INDICES, FOURTY_INDICES_INC

PARTIALLY_SUPERVISED = ["voc", "nvoc", "40_classes_inc"]


def get_gt_masks(proposals, size, device, train):
    """
    Takes ground-truth mask labels saved as PolygonMask and uses bounding boxes
    predicted by the model to crop and return pixelwise ground-truth labels.
    :param proposals: Detectron2 instances
    :param size: Size of the ground-truth masks
    :param device: Pytorch device
    :param train: During training we use proposal boxes created by the RPN. During
                  testing we use the refined boxes predicted by the box head.
    :return: Pixelwise ground-truth
    """
    gt_masks = []
    for prop in proposals:
        if train:
            boxes = prop.proposal_boxes.tensor
        else:
            boxes = prop.pred_boxes.tensor
        gt_per_image = prop.gt_masks.crop_and_resize(boxes, size).to(dtype=torch.float32).to(device=device)
        gt_masks.append(gt_per_image[:, None, :, :])
    return cat(gt_masks, dim=0)


def mask_loss_ps(pred_mask_logits, instances, vis_period=0, training_mode=''):
    """
    Copyright (c) Facebook, Inc. and its affiliates.
    Adapted Detectron2 function.

    :param pred_mask_logits: Predicted logits by the model
    :param instances: Detectron2 instances
    :param vis_period: Determines the iterations where predictions are visualized
    :param training_mode: Determines the supervised subset of mask labels ('voc', 'nvoc',
                          '40_classes_inc' or ''). `''` signifies training on all classes.
                          All instances not of the supervised set of classes are masked.
                          This way the model does not `see` the respective ground-truth
                          labels and no information of the novel classes is backpropagated.
    :return: Pixelwise binary cross entropy loss reduced with mean.
    """
    mask_side_len = pred_mask_logits.size(2)
    assert pred_mask_logits.size(2) == pred_mask_logits.size(3), "Mask prediction must be square!"
    assert training_mode in PARTIALLY_SUPERVISED + [''], \
        "partially_supervised is not 'voc', 'nvoc', '40_classes_inc' or '' (empty)"

    gt_masks = []
    global_ps_mask = []
    for instances_per_image in instances:
        if len(instances_per_image) == 0:
            continue

        gt_masks_per_image = instances_per_image.gt_masks.crop_and_resize(
            instances_per_image.proposal_boxes.tensor, mask_side_len
        ).to(device=pred_mask_logits.device)

        if training_mode in PARTIALLY_SUPERVISED:
            gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)

            ps_mask = get_ps_mask(gt_classes_per_image, training_mode)
            global_ps_mask.extend(ps_mask)

        # A tensor of shape (N, M, M), N=#instances in the image; M=mask_side_len
        gt_masks.append(gt_masks_per_image)

    gt_masks = cat(gt_masks, dim=0)
    if training_mode in PARTIALLY_SUPERVISED:
        pred_mask_logits = pred_mask_logits[global_ps_mask]
        gt_masks = gt_masks[global_ps_mask]

    if len(gt_masks) == 0 or len(pred_mask_logits) == 0:
        return pred_mask_logits.sum() * 0

    pred_mask_logits = pred_mask_logits[:, 0]

    if gt_masks.dtype == torch.bool:
        gt_masks_bool = gt_masks
    else:
        # Here we allow gt_masks to be float as well (depend on the implementation of rasterize())
        gt_masks_bool = gt_masks > 0.5
    gt_masks = gt_masks.to(dtype=torch.float32)

    # Log the training accuracy (using gt classes and 0.5 threshold)
    mask_incorrect = (pred_mask_logits > 0.0) != gt_masks_bool
    mask_accuracy = 1 - (mask_incorrect.sum().item() / max(mask_incorrect.numel(), 1.0))
    num_positive = gt_masks_bool.sum().item()
    false_positive = (mask_incorrect & ~gt_masks_bool).sum().item() / max(
        gt_masks_bool.numel() - num_positive, 1.0
    )
    false_negative = (mask_incorrect & gt_masks_bool).sum().item() / max(num_positive, 1.0)

    storage = get_event_storage()
    storage.put_scalar("mask/accuracy", mask_accuracy)
    storage.put_scalar("mask/false_positive", false_positive)
    storage.put_scalar("mask/false_negative", false_negative)
    if vis_period > 0 and storage.iter % vis_period == 0:
        pred_masks = pred_mask_logits.sigmoid()
        vis_masks = torch.cat([pred_masks, gt_masks], dim=2)
        name = "Left: mask prediction;   Right: mask GT"
        for idx, vis_mask in enumerate(vis_masks):
            vis_mask = torch.stack([vis_mask] * 3, dim=0)
            storage.put_image(name + f" ({idx})", vis_mask)

    mask_loss = F.binary_cross_entropy_with_logits(pred_mask_logits, gt_masks, reduction="mean")
    return mask_loss


def get_ps_mask(gt_classes_per_image, training_mode):
    """
    For the specific training mode it returns a binary mask for each proposal stating if its
    a supervised or non-supervised class. True= supervised, False=unsupervised.
    :param gt_classes_per_image: Ground truth classes for each images.
    :param training_mode: decides the supervised classes.
    :return: Binary list masking the unsupervised classes.
    """
    if training_mode == 'voc':
        return [gt_c in VOC_INDICES for gt_c in gt_classes_per_image]
    elif training_mode == 'nvoc':
        return [gt_c not in VOC_INDICES for gt_c in gt_classes_per_image]
    elif training_mode == '40_classes_inc':
        return [gt_c in FOURTY_INDICES_INC for gt_c in gt_classes_per_image]
