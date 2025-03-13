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
# from mm.segmentation.src.models.dinov2.vit_dinov2 import DinoVisionBackbone
from mm.segmentation.src.models.dinov2.dino_v2 import DinoVisionTransformer
from mm.segmentation.src.models.gcnet.gcnethead import GCNetHead 
from mm.segmentation.src.models.gcnet.gcnet import GCNet
from mm.segmentation.utils.functions import add_params_to_args
from mm.segmentation.configs.models.mask2former import backbone_weights_map
from mm.segmentation.configs.models.cosnet import backbone_weights_map as cosnet_backbone_weights_map
from mm.segmentation.configs.models.deeplabv3plus import backbone_weights_map as dlabv3plus_backbone_weights_map
from mm.segmentation.configs.models.pidnet import backbone_weights_map as pidnet_backbone_weights_map
from mm.segmentation.configs.models.gcnet import backbone_weights_map as pidnet_backbone_weights_map
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


def get_backbone_weights_map(model_name):
    if model_name == 'mask2former':
        return backbone_weights_map
    elif model_name == 'cosnet':
        return cosnet_backbone_weights_map
    elif model_name == 'deeplabv3plus':
        return dlabv3plus_backbone_weights_map
    elif model_name == 'pidnet':
        return pidnet_backbone_weights_map
    else:
        raise NotImplementedError(f'[ERROR] There is no such model name for backbone-weights-map: {model_name}')

def main():
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, args.args_filename)

    if 'dinov2' not in args.model:       
        args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, get_backbone_weights_map(args.model)[args.backbone], 'pth')

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
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
    cfg.model_wrapper_cfg=dict(
        type='MMDistributedDataParallel',
        find_unused_parameters=True
    )

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

def mask2former():
      
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/train.yaml')

    from datetime import datetime 
    now = datetime.now()
    output_dir = '/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/mask2former/outputs/SEGMENTATION'
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
    args.backbone = 'swin-s'
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

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
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


def cosnet():
      
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/train.yaml')

    from datetime import datetime 
    now = datetime.now()
    output_dir = '/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/cosnet/outputs/SEGMENTATION'
    input_dir = "/DeepLearning/_athena_tests/datasets/polygon2/split_dataset"
    classes = ['background', 'line', 'stabbed']

    rois = [[220, 60, 1340, 828]]
    patch = {
        "use_patch": True,
        "include_point_positive": True,
        "centric": False,
        "sliding": True,
        "width": 1120,
        "height": 768,
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
    
    args.model= 'cosnet'
    args.backbone = 'upernet-r50'
    args.height = 768
    args.width = 1120
    args.frozen_stages = 3
    
    args.rois = rois
    args.patch = patch

    args.batch_size = 2
    args.max_iters = 100
    args.val_interval = 50
    
    args.custom_hooks['visualize_val']['output_dir'] = val_dir
    args.custom_hooks['before_train']['debug_dataloader']['output_dir'] = debug_dir
    args.custom_hooks['aiv']['logging']['output_dir'] = logs_dir
    args.custom_hooks['checkpoint']['output_dir'] = weights_dir
    
    args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, cosnet_backbone_weights_map[args.backbone], 'pth')

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
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

def deeplabv3plus():
      
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/train.yaml')

    from datetime import datetime 
    now = datetime.now()
    output_dir = '/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/deeplabv3plus/outputs/SEGMENTATION'
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
    
    
    output_dir = osp.join(output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}', 'train')     
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
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
    
    args.model= 'deeplabv3plus'
    args.backbone = 'r101-d8'
    args.height = 256
    args.width = 512
    args.frozen_stages = 3
    
    args.rois = rois
    args.patch = patch

    args.batch_size = 2
    args.max_iters = 100
    args.val_interval = 50
    
    args.custom_hooks['visualize_val']['output_dir'] = val_dir
    args.custom_hooks['before_train']['debug_dataloader']['output_dir'] = debug_dir
    args.custom_hooks['aiv']['logging']['output_dir'] = logs_dir
    args.custom_hooks['checkpoint']['output_dir'] = weights_dir
    
    args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, dlabv3plus_backbone_weights_map[args.backbone], 'pth')

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
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

def pidnet():
      
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/train.yaml')

    from datetime import datetime 
    now = datetime.now()
    output_dir = '/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/pidnet/outputs/SEGMENTATION'
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
    
    
    output_dir = osp.join(output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}', 'train')     
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
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
    
    args.model= 'pidnet'
    args.backbone = 'l'
    args.height = 256
    args.width = 512
    args.frozen_stages = -1
    
    args.amp = False
    
    args.rois = rois
    args.patch = patch

    args.batch_size = 2
    args.max_iters = 100
    args.val_interval = 50
    
    args.custom_hooks['visualize_val']['output_dir'] = val_dir
    args.custom_hooks['before_train']['debug_dataloader']['output_dir'] = debug_dir
    args.custom_hooks['aiv']['logging']['output_dir'] = logs_dir
    args.custom_hooks['checkpoint']['output_dir'] = weights_dir
    
    args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, pidnet_backbone_weights_map[args.backbone], 'pth')

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
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
    
def dinov2(): # dinov2
          
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/train.yaml')

    from datetime import datetime 
    now = datetime.now()
    output_dir = '/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/dinov2/outputs/SEGMENTATION'
    input_dir = "/DeepLearning/_athena_tests/datasets/polygon2/split_dataset"
    classes = ['background', 'line', 'stabbed']

    rois = [[220, 60, 1340, 828]]
    patch = {
        "use_patch": True,
        "include_point_positive": True,
        "centric": False,
        "sliding": True,
        "width": 448,
        "height": 224,
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
    
    
    output_dir = osp.join(output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}', 'train')     
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
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

    
    args.model= 'dinov2'
    # args.backbone = 'vit-l-14'
    # args.backbone = 'vit-b-14'
    args.backbone = 'vit-s-14'
    args.height = 224
    args.width = 448
    args.frozen_stages = -1
    
    args.amp = False
    
    args.rois = rois
    args.patch = patch

    args.batch_size = 2
    args.max_iters = 100
    args.val_interval = 50
    
    args.custom_hooks['visualize_val']['output_dir'] = val_dir
    args.custom_hooks['before_train']['debug_dataloader']['output_dir'] = debug_dir
    args.custom_hooks['aiv']['logging']['output_dir'] = logs_dir
    args.custom_hooks['checkpoint']['output_dir'] = weights_dir
    
    # args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, dinov2_backbone_weights_map[args.backbone], 'pth')

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
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

def gcnet():  
    # set config =======================================================================================================
    args = parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/train.yaml')

    from datetime import datetime 
    now = datetime.now()
    output_dir = '/DeepLearning/etc/_athena_tests/recipes/agent/segmentation/mmseg/train_unit/gcnet/outputs/SEGMENTATION'
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
    
    
    output_dir = osp.join(output_dir, f'{now.month}_{now.day}_{now.hour}_{now.minute}_{now.second}', 'train')     
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
        
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

    
    args.model= 'gcnet'
    args.backbone = 's'
    args.height = 256
    args.width = 512
    args.frozen_stages = -1
    
    args.amp = False
    
    args.rois = rois
    args.patch = patch

    args.batch_size = 2
    args.max_iters = 100
    args.val_interval = 50
    
    args.custom_hooks['visualize_val']['output_dir'] = val_dir
    args.custom_hooks['before_train']['debug_dataloader']['output_dir'] = debug_dir
    args.custom_hooks['aiv']['logging']['output_dir'] = logs_dir
    args.custom_hooks['checkpoint']['output_dir'] = weights_dir
    
    # args.load_from = get_weights_from_nexus('segmentation', 'mmseg', args.model, dinov2_backbone_weights_map[args.backbone], 'pth')

    config_file = ROOT / f'segmentation/configs/models/{args.model}/{args.model}_{args.backbone}.py'
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
    main()
    # mask2former()
    # cosnet()
    # deeplabv3plus()
    # pidnet()
    # dinov2()
    # gcnet()