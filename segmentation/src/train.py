import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

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

def main():
    # set config =======================================================================================================
    args = parse_args()

    config_file = ROOT / 'segmentation/configs/models/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2.py'
    amp = True
    if args.cfg_options is None:
        cfg_options = {'load_from': '/HDD/weights/mmseg/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth',
                   'launcher': args.launcher, 
                   'resume': False,
                   'work_dir': '/HDD/datasets/projects/LX/24.12.12/outputs/mm'
            }
    else:
        cfg_options = args.cfg_options
        
    cfg = Config.fromfile(config_file)
    cfg.merge_from_dict(cfg_options)
    
    if amp is True:
        optim_wrapper = cfg.optim_wrapper.type
        if optim_wrapper == 'AmpOptimWrapper':
            print_log(
                'AMP training is already enabled in your config.',
                logger='current',
                level=logging.WARNING)
        else:
            assert optim_wrapper == 'OptimWrapper', (
                '`--amp` is only supported when the optimizer wrapper type is '
                f'`OptimWrapper` but got {optim_wrapper}.')
            cfg.optim_wrapper.type = 'AmpOptimWrapper'
            cfg.optim_wrapper.loss_scale = 'dynamic'

    # ================================================================================================================

    # set crop-size/model-size =================================================================================
    height = 512
    width = 512
    new_crop_size = (height, width)
    cfg.crop_size = new_crop_size 
    cfg.data_preprocessor.size = new_crop_size
    cfg.model.data_preprocessor = cfg.data_preprocessor
    if 'train_pipeline' in cfg and isinstance(cfg.train_pipeline, list):
        for pipeline in cfg.train_pipeline:
            if pipeline.get('type') == 'RandomCrop':
                pipeline['crop_size'] = tuple(new_crop_size)

    if 'val_pipeline' in cfg and isinstance(cfg.val_pipeline, list):
        for pipeline in cfg.val_pipeline:
            if pipeline.get('type') == 'RandomCrop':
                pipeline['crop_size'] = tuple(new_crop_size)
    # =============================================================================================================
    
    num_classes = 150
    # set num_classes =================================================================================
    cfg.num_classes = num_classes 
    if 'model' in cfg:
        if cfg.model.get('type') == 'EncoderDecoder':
            if 'decode_head' in cfg.model and 'num_classes' in cfg.model.decode_head:
                cfg.model.decode_head.num_classes = num_classes
            if 'loss_cls' in cfg.model and 'num_classes' in cfg.model.loss_dict:
                cfg.model.loss_dict.loss_weight = [1.0] * num_classes + [0.1]
    # =============================================================================================================
    
    max_iters = 160 
    val_interval = 50
    checkpoint_interval = 50
    # set num_classes =================================================================================
    if cfg.train_cfg.get('type') == 'IterBasedTrainLoop':
        cfg.train_cfg.max_iters = max_iters
        cfg.train_cfg.val_interval = val_interval
        
    if 'param_scheduler' in cfg and isinstance(cfg.param_scheduler, list):
        for scheduler in cfg.param_scheduler:
            if scheduler.get('type') == 'PolyLR':
                scheduler['end'] = max_iters
        
    
    if 'checkpoint' in cfg.default_hooks:
        cfg.default_hooks.checkpoint.interval = checkpoint_interval
    # =============================================================================================================
    
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # start training
    runner.train()


if __name__ == '__main__':
    main()