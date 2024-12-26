import argparse
import logging
import os
import os.path as osp
import yaml

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from segmentation.src.datasets.mask_dataset import MaskDataset
from segmentation.utils.config import ConfigManager
from segmentation.src.runners import RunnerV1

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    return args

def add_params_to_args(args, params_file):
    with open(params_file) as yf:
        params = yaml.load(yf, Loader=yaml.FullLoader)

    for key, value in params.items():
        setattr(args, key, value)


def main():
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'params/mask.yaml')

    config_file = ROOT / '../configs/models/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2.py'
    config_manager = ConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_schedule_config(args.max_iters, args.val_interval, args.checkpoint_interval)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, args.classes, args.batch_size, args.width, args.height)
    cfg = config_manager.cfg

    # ================================================================================================================
    if 'runner_type' not in cfg:
        runner = RunnerV1.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()