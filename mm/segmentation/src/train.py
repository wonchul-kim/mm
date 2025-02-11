import argparse
import logging
import os
import os.path as osp

import os

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from mm.utils.weights import get_weights_from_nexus
from mm.segmentation.src.datasets.mask_dataset import MaskDataset
from mm.segmentation.utils.hooks import VisualizeVal
from mm.segmentation.utils.metrics import IoUMetricV2
from mm.segmentation.utils.config import TrainConfigManager
from mm.segmentation.src.runners import RunnerV1
from mm.segmentation.utils.functions import add_params_to_args
from mm.segmentation.src.models.mask2former import backbone_weights_map
import mm.segmentation.utils.transforms.loading
import mm.segmentation.src.loops

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

def parse_args():
    parser = argparse.ArgumentParser(description='Train a segmentor')
    parser.add_argument('--args-filename')
    
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
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, args.args_filename)

    args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, backbone_weights_map[args.backbone], 'pth')

    config_file = ROOT / f'../configs/models/mask2former/{args.model}_{args.backbone}_8xb2.py'
    config_manager = TrainConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_schedule_config(args.max_iters, args.val_interval)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, args.classes, args.batch_size, args.width, args.height)
    config_manager.manage_default_hooks_config(args.default_hooks)
    # config_manager.manage_dataloader_config(args.vis_dataloader_ratio)
    config_manager.manage_custom_hooks_config(args.custom_hooks)
    cfg = config_manager.cfg

    # ================================================================================================================
    if 'runner_type' not in cfg:
        # runner = RunnerV1.from_cfg(cfg)
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()



def main2():
      
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/train.yaml')

    from datetime import datetime 
    now = datetime.now()
    output_dir = '/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/mm_m2f/train'
    input_dir = "/DeepLearning/_athena_tests/datasets/polygon2/split_dataset"
    classes = ['background', 'line', 'stabbed']

    rois = [[220, 60, 1340, 828]]
    patch = {
        "use_patch": True,
        "include_point_positive": True,
        "centric": False,
        "sliding": True,
        "width": 512,
        "height": 256,
        "overlap_ratio": 0.2,
        "num_involved_pixel": 10,
        "sliding_bg_ratio": 0,
        "bg_ratio_by_image": 0,
        "bg_start_train_epoch_by_image": 0,
        "bg_start_val_epoch_by_image": 0,
        "translate": 0,
        "translate_range_width": 0,
        "translate_range_height": 0,
    }
    
    
    output_dir = osp.join(output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}')     
    if not osp.exists(output_dir):
        os.mkdir(output_dir)
        
    val_dir = osp.join(output_dir, 'val')
    os.mkdir(val_dir)
    
    debug_dir = osp.join(output_dir, 'debug')
    os.mkdir(debug_dir)
    
    logs_dir = osp.join(output_dir, 'logs')
    os.mkdir(logs_dir)
    
    weights_dir = osp.join(output_dir, 'weights')
    os.mkdir(weights_dir)
    
    args.output_dir = output_dir
    args.data_root = input_dir
    args.classes = classes
    args.num_classes = len(classes)
    
    args.model= 'mask2former'
    args.backbone = 'swin-t'
    args.height = 256
    args.width = 512
    
    args.rois = rois
    args.patch = patch

    args.epochs = 0
    args.max_iters = 100
    args.val_interval = 50
    
    args.custom_hooks['visualize_val']['output_dir'] = val_dir
    args.custom_hooks['before_train']['debug_dataloader']['output_dir'] = debug_dir
    args.custom_hooks['aiv']['logging']['output_dir'] = logs_dir
    args.custom_hooks['checkpoint']['output_dir'] = weights_dir
    
    args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, backbone_weights_map[args.backbone], 'pth')

    config_file = ROOT / f'segmentation/configs/models/mask2former/{args.model}_{args.backbone}_8xb2.py'
    config_manager = TrainConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_schedule_config(args.max_iters, args.val_interval)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, 
                                         args.classes, args.batch_size, args.width, args.height,
                                         args.rois, args.patch)
    config_manager.manage_default_hooks_config(args.default_hooks)
    # config_manager.manage_dataloader_config(args.vis_dataloader_ratio)
    config_manager.manage_custom_hooks_config(args.custom_hooks)
    cfg = config_manager.cfg

    # ================================================================================================================
    if 'runner_type' not in cfg:
        # runner = RunnerV1.from_cfg(cfg)
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    # main()
    main2()