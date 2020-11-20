import os
import shutil

import torch

from fvcore.common.file_io import PathManager

from detectron2.utils import comm
from detectron2.utils.env import seed_all_rng
from detectron2.utils.logger import setup_logger
from detectron2.utils.collect_env import collect_env_info
from detectron2.config import get_cfg
from detectron2.config import CfgNode as CN
from detectron2.engine.defaults import default_setup


def add_opmask_cfg(cfg):
    """
    Additional OPMask parameters to the config object
    """

    cfg.MODEL.ROI_MASK_HEAD.NUM_CONV_UP = 1

    cfg.EXP = CN()
    cfg.EXP.TRAINER = "General"  # Determines the Trainer used.
    cfg.EXP.DATASET = "coco"
    cfg.EXP.PS = ''  # "voc", "nvoc", "10_classes", "30_classes", "40_classes_inc"
    cfg.EXP.PRINT_PERIOD = 20  # For debugging purposes


def overall_setup(args):
    cfg = get_cfg()  # obtain detectron2's default config
    add_opmask_cfg(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    setup_paths(cfg, args)
    cfg.freeze()
    default_setup(cfg, args)
    if not (args.eval_only or args.resume):
        save_exp_setup(args, cfg)
    return cfg


def setup_paths(cfg, args):
    folder_name = args.folder_name
    dataset_name = cfg.EXP.DATASET.lower()
    experiment_folder = f"{dataset_name}_{args.exp_id}"
    base_path = os.path.join(f"./output/{folder_name}", experiment_folder)
    cfg.OUTPUT_DIR = base_path


def create_experiment_directory(base_path, eval_only, resume=False):
    if eval_only or resume:
        if os.path.exists(base_path):
            pass  # If we do evaluation we dont want to destroy our saved models
        else:  # in case we loaded a pretrained model
            os.makedirs(base_path)
            os.makedirs(os.path.join(base_path, "inference"))
    else:
        # Zip old experiment to not destroy it right away
        if os.path.exists(base_path):
            shutil.make_archive(base_path, 'zip', base_path)
            shutil.rmtree(base_path)
        os.makedirs(base_path)
        os.makedirs(os.path.join(base_path, "tensorboard"))
        os.makedirs(os.path.join(base_path, "inference"))
        os.makedirs(os.path.join(base_path, "models"))


def save_exp_setup(args, cfg):
    """
    Detectron2 overwrites saved configs when evaluating the same model.
    This saves the original configs for sanity checks later.
    """
    base_path = cfg.OUTPUT_DIR
    with open(os.path.join(base_path, 'experiment_configs.txt'), 'w') as f:
        print("Command line arguments: \n", file=f)
        print(args, "\n", file=f)
        print("Detectron arguments arguments: \n", file=f)
        print(cfg, file=f)


def default_setup(cfg, args):
    """
    Perform some basic common setups at the beginning of a job, including:

    1. Set up the detectron2 logger
    2. Log basic information about environment, cmdline arguments, and config
    3. Backup the config to the output directory

    Args:
        cfg (CfgNode): the full config to be used
        args (argparse.NameSpace): the command line arguments to be logged
    """
    output_dir = cfg.OUTPUT_DIR
    if comm.is_main_process() and output_dir:
        create_experiment_directory(output_dir, args.eval_only, args.resume)

    rank = comm.get_rank()
    setup_logger(output_dir, distributed_rank=rank, name="fvcore")
    logger = setup_logger(output_dir, distributed_rank=rank)

    logger.info("Rank of current process: {}. World size: {}".format(rank, comm.get_world_size()))
    logger.info("Environment info:\n" + collect_env_info())

    logger.info("Command line arguments: " + str(args))
    if hasattr(args, "config_file") and args.config_file != "":
        logger.info(
            "Contents of args.config_file={}:\n{}".format(
                args.config_file, PathManager.open(args.config_file, "r").read()
            )
        )

    logger.info("Running with full config:\n{}".format(cfg))
    if comm.is_main_process() and output_dir:
        # Note: some of our scripts may expect the existence of
        # config.yaml in output directory
        path = os.path.join(output_dir, "config.yaml")
        with PathManager.open(path, "w") as f:
            f.write(cfg.dump())
        logger.info("Full config saved to {}".format(os.path.abspath(path)))

    # make sure each worker has a different, yet deterministic seed if specified
    seed_all_rng(None if cfg.SEED < 0 else cfg.SEED + rank)

    # cudnn benchmark has large overhead. It shouldn't be used considering the small size of
    # typical validation set.
    if not (hasattr(args, "eval_only") and args.eval_only):
        torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK
