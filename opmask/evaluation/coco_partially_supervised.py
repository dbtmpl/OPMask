import os
import json
import copy
import itertools

import numpy as np
from fvcore.common.file_io import PathManager
from pycocotools.cocoeval import COCOeval
from detectron2.evaluation import COCOEvaluator

from ..utils.class_splits import VOC_IDS, VOC_INDICES, NON_VOC_IDS, NON_VOC_INDICES


class PartiallySupervisedEvaluator(COCOEvaluator):

    def _eval_predictions(self, tasks, predictions):
        """
        Evaluate predictions on the given tasks.
        Fill self._results with the metrics of the tasks.
        """
        self._logger.info("Preparing results for COCO format ...")
        coco_results = list(itertools.chain(*[x["instances"] for x in predictions]))

        # unmap the category ids for COCO
        if hasattr(self._metadata, "thing_dataset_id_to_contiguous_id"):
            reverse_id_mapping = {
                v: k for k, v in self._metadata.thing_dataset_id_to_contiguous_id.items()
            }
            for result in coco_results:
                category_id = result["category_id"]
                assert (
                        category_id in reverse_id_mapping
                ), "A prediction has category_id={}, which is not available in the dataset.".format(
                    category_id
                )
                result["category_id"] = reverse_id_mapping[category_id]

        if self._output_dir:
            file_path = os.path.join(self._output_dir, "coco_instances_results.json")
            self._logger.info("Saving results to {}".format(file_path))
            with PathManager.open(file_path, "w") as f:
                f.write(json.dumps(coco_results))
                f.flush()

        if not self._do_evaluation:
            self._logger.info("Annotations are not available for evaluation.")
            return

        self._logger.info("Evaluating predictions ...")
        for i, task in enumerate(list(sorted(tasks)) + ['segm_voc', 'segm_nvoc']):

            if task in ['segm_voc', 'segm_nvoc']:
                iou_type, ps_mode = task.split('_')
            else:
                iou_type = task
                ps_mode = None

            print('psmode', ps_mode, iou_type)
            coco_eval = (
                _evaluate_predictions_on_coco_ps(
                    self._coco_api, coco_results, iou_type, ps_mode, kpt_oks_sigmas=self._kpt_oks_sigmas
                )
                if len(coco_results) > 0
                else None  # cocoapi does not handle empty results very well
            )

            if ps_mode == 'voc':
                class_names = list(np.asarray(self._metadata.get("thing_classes"))[VOC_INDICES])
            elif ps_mode == 'nvoc':
                class_names = list(np.asarray(self._metadata.get("thing_classes"))[NON_VOC_INDICES])
            else:
                class_names = self._metadata.get("thing_classes")

            res = self._derive_coco_results(
                coco_eval, iou_type, class_names=class_names
            )
            self._results[task] = res


def _evaluate_predictions_on_coco_ps(coco_gt, coco_results, iou_type, ps_mode, kpt_oks_sigmas=None):
    """
    Evaluate the coco results using COCOEval API. Adapted to allow evaluation on different class splits.
    """
    assert len(coco_results) > 0

    if iou_type == "segm":
        coco_results = copy.deepcopy(coco_results)
        # When evaluating mask AP, if the results contain bbox, cocoapi will
        # use the box area as the area of the instance, instead of the mask area.
        # This leads to a different definition of small/medium/large.
        # We remove the bbox field to let mask AP use mask area.
        for c in coco_results:
            c.pop("bbox", None)

    coco_dt = coco_gt.loadRes(coco_results)
    coco_eval = COCOeval(coco_gt, coco_dt, iou_type)
    if ps_mode == 'voc':
        coco_eval.params.catIds = VOC_IDS
    elif ps_mode == 'nvoc':
        coco_eval.params.catIds = NON_VOC_IDS

    # Use the COCO default keypoint OKS sigmas unless overrides are specified
    if kpt_oks_sigmas:
        coco_eval.params.kpt_oks_sigmas = np.array(kpt_oks_sigmas)

    if iou_type == "keypoints":
        num_keypoints = len(coco_results[0]["keypoints"]) // 3
        assert len(coco_eval.params.kpt_oks_sigmas) == num_keypoints, (
            "[COCOEvaluator] The length of cfg.TEST.KEYPOINT_OKS_SIGMAS (default: 17) "
            "must be equal to the number of keypoints. However the prediction has {} "
            "keypoints! For more information please refer to "
            "http://cocodataset.org/#keypoints-eval.".format(num_keypoints)
        )

    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    return coco_eval
