# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner
from mm.segmentation.src.models.gcnet.gcnethead import GCNetHead 
from mm.segmentation.src.models.gcnet.gcnet import GCNet
from mm.segmentation.src.datasets.mask_dataset import MaskDataset
from mm.segmentation.utils.hooks import VisualizeTest
from mm.segmentation.utils.metrics import IoUMetricV2
from mm.segmentation.utils.config import TestConfigManager
from mm.utils.functions import add_params_to_args, increment_path
import mm.segmentation.utils.transforms
import mm.segmentation.src.loops
import mm.segmentation.utils.losses


from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]


# TODO: support fuse_conv_bn, visualization, and format_only
def parse_args():
    
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
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
    
    args = parse_args()
    add_params_to_args(args, args.args_filename)
    
    config_file = ROOT / f'segmentation/configs/models/mask2former/{args.model}_{args.backbone}.py'
    config_manager = TestConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, 
                                         args.seg_map_suffix, args.classes, 
                                         args.batch_size, args.width, args.height,
                                         args.rois, args.patch)
    config_manager.manage_custom_test_hooks_config(args.custom_hooks)

    cfg = config_manager.cfg

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    runner = Runner.from_cfg(cfg)
    runner.test()

def main1():
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/test.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_mask2former_swin-s_w1120_h768.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/mr_infra/test_bottom_segformer_b2_unfrozen_w512_h512.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_segformer_b2_unfrozen_w1120_h768.yaml')
    add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_segformer_b2_unfrozen_w1120_h768_tta.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_segnext_w1120_h768.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_segnext_w1120_h768_tta.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_custom_deeplabv3plus_w1120_h768.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_lps_w1120_h768.yaml')
    # add_params_to_args(args, ROOT / 'segmentation/data/projects/tenneco/test_outer_gcnet_w1120_h768.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_lps_512_epochs200_unit.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_segman_epochs200.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_m2f_epochs100.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_sam2unet_epochs300.yaml')
    # add_params_to_args(args, ROOT /'segmentation/data/projects/tenneco/test_outer_mask2former_epochs100.yaml')
    
    args.create_output_dirs = True
    
    if args.create_output_dirs:
        from mm.utils.functions import create_output_dirs
        create_output_dirs(args, 'test')
        print(f"CREATED output-dirs: {args.output_dir}")
    
    args.load_from = args.weights
    args.data_root = args.input_dir
    args.num_classes = len(args.classes)
    
    args.custom_hooks['visualize_test']['output_dir'] = osp.join(args.output_dir, 'vis')

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
    config_manager = TestConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, 
                                         args.classes, args.batch_size, args.width, args.height,
                                         args.rois, args.patch)
    config_manager.manage_custom_test_hooks_config(args.custom_hooks)

    cfg = config_manager.cfg

    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    # main()
    main1()
    # main2()