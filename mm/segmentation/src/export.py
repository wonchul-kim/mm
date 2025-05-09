# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mm.segmentation.src.datasets.mask_dataset import MaskDataset
from mm.segmentation.utils.hooks import VisualizeVal
from mm.segmentation.utils.metrics import IoUMetricV2
from mm.segmentation.utils.config import TrainConfigManager
from mm.segmentation.src.runners import RunnerV1

from mmdeploy.utils import get_ir_config
from mm.utils.functions import add_params_to_args
from mm.segmentation.utils.export.torch2onnx import torch2onnx
from mm.segmentation.utils.config.export_config_manager import ExportConfigManager

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]


def to_list(s):
    if isinstance(s, int):  
        return [s]
    elif isinstance(s, str): 
        return [int(x.strip()) for x in s.split(',')]
    else:
        raise ValueError("Input must be an integer or a comma-separated string")

def parse_args():
    parser = argparse.ArgumentParser(description='MMSeg export a model')
    parser.add_argument('--args-filename')

    return parser.parse_args()

def main():
    args = parse_args()
    add_params_to_args(args, args.args_filename)
    
    # parser = argparse.ArgumentParser(description='MMSeg export a model')
    # args = parser.parse_args()
    # add_params_to_args(args, ROOT / 'configs/recipe/export.yaml')
    
    config_manager = ExportConfigManager()
    config_manager.build(args, str(ROOT / f'segmentation/configs/_base_/onnx_config.py'))

    torch2onnx(
        config_manager.cfg['model_inputs'],
        args.work_dir,
        get_ir_config(config_manager.cfg)['save_file'],
        deploy_cfg=config_manager.cfg,
        model_cfg=args.model_cfg,
        model_checkpoint=args.checkpoint,
        device=args.device)

def main2():
    
    # output_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/m2f_epochs100/export'
    # model_cfg =  '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/m2f_epochs100/train/mask2former_swin-s.py'
    # checkpoint = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/m2f_epochs100/train/weights/best_mIoU_iter_47800.pth'
    # output_dir = "/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/pidnet_epochs100/export"
    # model_cfg = "/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/pidnet_epochs100/train/pidnet_l.py" 
    # checkpoint = "/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/pidnet_epochs100/train/weights/best_mIoU_iter_23800.pth"
    
    output_dir = "/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/sam2_epochs300/export"
    model_cfg = "/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/sam2_epochs300/train/sam2_s.py" 
    checkpoint = "/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/sam2_epochs300/train/weights/best_mIoU_iter_134318.pth"
    
    
    parser = argparse.ArgumentParser(description='MMSeg export a model')
    args = parser.parse_args()
    add_params_to_args(args, ROOT / 'segmentation/configs/recipe/export.yaml')
    args.model_cfg = model_cfg 
    args.checkpoint = checkpoint 
    args.work_dir = output_dir
    
    # export
    args.device = 'cuda'
    args.batch_size = 1
    args.onnx_config["opset_version"] = 14
    
    # model
    # args.model = 'mask2former'
    # args.backbone = 'swin-s'
    # args.model = 'pidnet'
    # args.backbone = 'l'
    args.model = 'sam2'
    args.backbone = 's'
    args.height = 768
    args.width = 1120
    
    args.tta = {'use': False, 'augs':{
                                        'HorizontalFlip': True,
                                        'VerticalFlip': True, 
                                        'Rotate': 90,
                                        'Translate': '100,100'
                            }
                }
    
    config_manager = ExportConfigManager()
    config_manager.build(args, str(ROOT / f'segmentation/configs/_base_/onnx_config.py'))
    cfg = config_manager.cfg
    
    torch2onnx(
        cfg['model_inputs'],
        args.work_dir,
        get_ir_config(cfg)['save_file'],
        deploy_cfg=cfg,
        model_cfg=args.model_cfg,
        model_checkpoint=args.checkpoint,
        device=args.device, 
        tta=cfg.tta)


if __name__ == '__main__':
    # main()
    main2()
