import argparse
import logging
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.logging import print_log
from mmengine.runner import Runner

from mmseg.registry import RUNNERS
from segmentation.src.datasets.mask_dataset import MaskDataset
from segmentation.utils.config import ConfigManager
from segmentation.src.runners import RunnerV1

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

    output_dir = '/HDD/datasets/projects/LX/24.12.12/outputs/mm/mask2former_swin-l-in22k'
    config_file = ROOT / 'segmentation/configs/models/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2.py'
    amp = True
    if args.cfg_options is None:
        cfg_options = {'load_from': '/HDD/weights/mmseg/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth',
                   'launcher': args.launcher, 
                   'resume': False,
                   'work_dir': output_dir
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
    height = 640
    width = 640
    new_crop_size = (height, width)
    num_classes = 2
    max_iters = 40000
    val_interval = 100
    checkpoint_interval = 500
    data_root = "/HDD/datasets/projects/LX/24.12.12/split_mask_patch_dataset"
    img_suffix='.png'
    seg_map_suffix='.png'
    classes = ('timber', 'screw')
    batch_size = 1
    config_manager = ConfigManager(cfg)
    config_manager.manage_model_config(num_classes, new_crop_size)
    config_manager.manage_schedule_config(max_iters, val_interval, checkpoint_interval)
    config_manager.manage_dataset_config(data_root, img_suffix, seg_map_suffix, classes, batch_size, new_crop_size)
    
    # ================================================================================================================
    if 'runner_type' not in cfg:
        # build the default runner
        # runner = Runner.from_cfg(cfg)
        runner = RunnerV1.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)

    # vis_dataloader = True
    # if vis_dataloader:
    #     from segmentation.utils.visualizers import vis_dataloader
    #     dataloader = runner.build_dataloader(cfg.train_dataloader)
    #     vis_dataloader(dataloader)
    
    # start training
    runner.train()


if __name__ == '__main__':
    main()