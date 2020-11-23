import torch

from torch.nn import functional as F


def process_cams_batch(cams_batch, instances, cam_res, is_train=True):
    """
    Use ground-truth (during training) or predicted class labels (during evaluation) to select the
    CAM slices corresponding to the primary class in the RoI. Afterwards, the CAMs are normalized
    to the range [0, 1] and resized to the desired spatial resolution.
    """
    if is_train:
        classes = torch.cat([x.gt_classes for x in instances], dim=0)
    else:
        classes = torch.cat([x.pred_classes for x in instances], dim=0)
    cams_batch = cams_batch[torch.arange(cams_batch.size(0)), classes][:, None, ...]
    return normalize_and_interpolate_batch(cams_batch, cam_res)


def normalize_and_interpolate_batch(cams_batch, cam_res):
    """
    Normalize and resize CAMs.
    """
    cams_batch = normalize_batch(cams_batch)
    return F.interpolate(cams_batch, scale_factor=(cam_res / cams_batch.size(2)), mode='bilinear')


def normalize_batch(cams_batch):
    """
    Classic min-max normalization
    """
    bs = cams_batch.size(0)
    cams_batch = cams_batch + 1e-4
    cam_mins = getattr(cams_batch.view(bs, -1).min(1), 'values').view(bs, 1, 1, 1)
    cam_maxs = getattr(cams_batch.view(bs, -1).max(1), 'values').view(bs, 1, 1, 1)
    return (cams_batch - cam_mins) / (cam_maxs - cam_mins)
