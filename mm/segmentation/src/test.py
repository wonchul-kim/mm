# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mm.segmentation.src.datasets.mask_dataset import MaskDataset
from mm.segmentation.utils.hooks import VisualizeTest
from mm.segmentation.utils.metrics import IoUMetricV2
from mm.segmentation.utils.config import TestConfigManager
from mm.segmentation.utils.functions import add_params_to_args, trigger_visualization_hook
import mm.segmentation.utils.transforms.loading
import mm.segmentation.src.loops

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

def increment_path(path, exist_ok=False, sep="", mkdir=False):
    from glob import glob
    import re 
    
    # Increment file or directory path, i.e. runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.
    path = Path(path)  # os-agnostic
    if path.exists() and not exist_ok:
        path, suffix = (
            (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        )
        dirs = glob(f"{path}{sep}*")  # similar paths
        matches = [re.search(rf"%s{sep}(\d+)" % path.stem, d) for d in dirs]
        i = [int(m.groups()[0]) for m in matches if m]  # indices
        n = max(i) + 1 if i else 2  # increment number
        path = Path(f"{path}{sep}{n}{suffix}")  # increment path
    if mkdir:
        path.mkdir(parents=True, exist_ok=True)  # make directory
    return path

def main():
    
    args = parse_args()
    add_params_to_args(args, args.args_filename)
    
    config_file = ROOT / f'segmentation/configs/models/mask2former/{args.model}_{args.backbone}_8xb2.py'
    config_manager = TestConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, args.classes, args.batch_size, args.width, args.height)
    config_manager.manage_custom_test_hooks_config(args.custom_hooks)

    cfg = config_manager.cfg

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    runner = Runner.from_cfg(cfg)
    runner.test()

def main2():
    output_dir = "/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/mm_m2f/test/exp"
    weights = "/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/mm_m2f/train/2_6_15_37_14/weights/iter_20000.pth"
    input_dir = "/DeepLearning/_athena_tests/datasets/polygon2/split_roi_patch_dataset/test"
    classes = ['background', 'line', 'stabbed']
    rois = [[]]
    patch = {
        "use_patch": False,
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

    
    parser = argparse.ArgumentParser(
        description='MMSeg test (and eval) a model')
    args = parser.parse_args()
    args.cfg_options = None
    args.launcher = 'none'
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/test.yaml')
    
    output_dir = str(increment_path(output_dir))
    args.output_dir = output_dir
    args.load_from = weights
    args.data_root = input_dir
    args.classes = classes
    args.num_classes = len(classes)
    
    args.model= 'mask2former'
    args.backbone = 'swin-s'
    args.height = 256
    args.width = 512
    
    args.rois = rois
    args.patch = patch
    
    args.custom_hooks['visualize_test']['output_dir'] = osp.join(output_dir, 'vis')
    args.custom_hooks['visualize_test']['contour_thres'] = 50
    args.custom_hooks['visualize_test']['annotate'] = True

    config_file = ROOT / f'segmentation/configs/models/mask2former/{args.model}_{args.backbone}_8xb2.py'
    config_manager = TestConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, 
                                         args.classes, args.batch_size, args.width, args.height,
                                         args.rois, args.patch)
    config_manager.manage_custom_test_hooks_config(args.custom_hooks)

    cfg = config_manager.cfg

    if args.tta:
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        cfg.tta_model.module = cfg.model
        cfg.model = cfg.tta_model

    runner = Runner.from_cfg(cfg)
    runner.test()

if __name__ == '__main__':
    # main()
    main2()
