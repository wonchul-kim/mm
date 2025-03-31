# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mm.segmentation.src.datasets.mask_dataset import MaskDataset
from mm.segmentation.utils.config import TestConfigManager
from mm.utils.functions import add_params_to_args, trigger_visualization_hook

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
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


def main():
    args = parse_args()
    add_params_to_args(args, ROOT / 'tests/test/params/labelme.yaml')

    config_file = ROOT / 'configs/models/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2.py'
    config_manager = TestConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, args.classes, args.batch_size, args.width, args.height)

    cfg = config_manager.cfg

    if args.show or cfg.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    # add output_dir in metric
    if args.out is not None:
        cfg.test_evaluator['output_dir'] = args.out
        cfg.test_evaluator['keep_results'] = True
        
    # build the runner from config
    runner = Runner.from_cfg(cfg)

    runner.model.to('cuda')

    # start testing
    runner.test()


if __name__ == '__main__':
    main()
