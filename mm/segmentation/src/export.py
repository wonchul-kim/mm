# Copyright (c) OpenMMLab. All rights reserved.
import argparse

from mm.segmentation.src.datasets.mask_dataset import MaskDataset
from mm.segmentation.utils.hooks import VisualizeVal
from mm.segmentation.utils.metrics import IoUMetricV2
from mm.segmentation.utils.config import TrainConfigManager
from mm.segmentation.src.runners import RunnerV1

from mmdeploy.utils import get_ir_config
from mm.segmentation.utils.functions import add_params_to_args
from mm.segmentation.utils.export.torch2onnx import torch2onnx
from mm.segmentation.utils.config.export_config_manager import ExportConfigManager

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

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
    config_manager.build(args, str(ROOT / f'configs/_base_/onnx_config.py'))

    torch2onnx(
        config_manager.cfg['model_inputs'],
        args.work_dir,
        get_ir_config(config_manager.cfg)['save_file'],
        deploy_cfg=config_manager.cfg,
        model_cfg=args.model_cfg,
        model_checkpoint=args.checkpoint,
        device=args.device)


if __name__ == '__main__':
    main()
