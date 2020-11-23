import os
import argparse

import detectron2.utils.comm as comm
from detectron2.engine import launch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import verify_results

from opmask.engine import GeneralTrainer
from opmask.utils.general import overall_setup

TRAINER = {
    "General": GeneralTrainer
}


def perform_eval(cfg, trainer):
    """
    Uses class methods of the trainer to build, load and evaluate a trained model.
    :param cfg: Namespace containing all OPMask configs.
    :param trainer: Trainer used for training the model.
    :return: Evaluation results.
    """
    model = trainer.build_model(cfg)
    DetectionCheckpointer(model, save_dir=os.path.join(cfg.OUTPUT_DIR, "models")).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    res = trainer.test(cfg, model)
    if comm.is_main_process():
        verify_results(cfg, res)
    return res


def train_or_eval(args, cfg, trainer):
    """
    Depending on the configs, training or evaluation is performed.
    """
    if args.eval_only:
        return perform_eval(cfg, trainer)

    trainer = trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def main(args):
    """
    Performs setup, choses trainer and starts training or evaluation.
    """
    cfg = overall_setup(args)
    trainer = TRAINER[cfg.EXP.TRAINER]
    return train_or_eval(args, cfg, trainer)


if __name__ == "__main__":
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid()) % 2 ** 14

    parser = argparse.ArgumentParser(description="General setup")

    # Experiment Housekeeping
    parser.add_argument('--folder-name', type=str, default="OPMask", help="Specify the folder the model is saved in")
    parser.add_argument('--config-file', default="", type=str, help='Path to the config file used')
    parser.add_argument("--eval-only", action="store_true", help="perform evaluation only")
    parser.add_argument("--resume", action="store_true", help="Resume training or start new")
    parser.add_argument('--exp-id', type=str, default="run_001", help="Run-id to distinguish experiments")
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
             "See config references at "
             "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    # Multi-GPU settings
    parser.add_argument('--num-gpus', type=int, default=1, help="number of GPUs to use")
    parser.add_argument("--dist-url", default="tcp://127.0.0.1:{}".format(port))
    parser.add_argument("--num-machines", type=int, default=1)
    parser.add_argument("--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)")

    args = parser.parse_args()

    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
