# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import logging
import os
import os.path as osp

from mm.segmentation.src.datasets.mask_dataset import MaskDataset
from mm.segmentation.utils.hooks import VisualizeVal
from mm.segmentation.utils.metrics import IoUMetricV2
from mm.segmentation.utils.config import TrainConfigManager
from mm.segmentation.src.runners import RunnerV1

from mmdeploy.utils import (get_ir_config, load_config)
from mm.utils.functions import add_params_to_args
from mm.segmentation.utils.export.torch2onnx import torch2onnx

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]

def to_list(s):
    if isinstance(s, int):  # If s is already an integer
        return [s]
    elif isinstance(s, str):  # If s is a string
        return [int(x.strip()) for x in s.split(',')]
    else:
        raise ValueError("Input must be an integer or a comma-separated string")


def main():
    parser = argparse.ArgumentParser(description='MMSeg export a model')
    args = parser.parse_args()
    add_params_to_args(args, ROOT / 'export/params/param.yaml')

    # args.batch_size = to_list(args.batch_size)
    # args.width = to_list(args.width)
    # args.height = to_list(args.height)
    
    # assert len(args.batch_size) == len(args.height), ValueError(f"[ERROR] the number of batch-size({len(args.batch_size)}) must be same to the number of height({len(args.height)})")
    # assert len(args.batch_size) == len(args.width), ValueError(f"[ERROR] the number of batch-size({len(args.batch_size)}) must be same to the number of width({len(args.width)})")
            
    deploy_cfg = load_config(args.deploy_cfg)[0]
    for key, val in args.codebase_config.items():
        deploy_cfg['codebase_config'][key] = val
    
    for key, val in args.onnx_config.items():
        deploy_cfg['onnx_config'][key] = val
        
    deploy_cfg['onnx_config']['input_shape'] = [args.width, args.height]
    deploy_cfg['onnx_config']['save_file'] = osp.join(args.work_dir, f'{args.model_name}_{args.backbone}_b{args.batch_size}_w{args.width}_h{args.height}')
        
    for key, val in args.backend_config.items():
        deploy_cfg['backend_config'][key] = val
        
    for idx in range(len(deploy_cfg['onnx_config']['output_names'])):
        deploy_cfg['backend_config']['model_inputs'][idx] = dict(
                input_shapes=dict(
                    input=dict(
                        min_shape=[args.batch_size, 3, args.height, args.width],
                        opt_shape=[args.batch_size, 3, args.height, args.width],
                        max_shape=[args.batch_size, 3, args.height, args.width])))
        # if len(args.batch_size) == 1:
        #     deploy_cfg['backend_config']['model_inputs'][idx] = dict(
        #         input_shapes=dict(
        #             input=dict(
        #                 min_shape=[args.batch_size[0], 3, args.height[0], args.width[0]],
        #                 opt_shape=[args.batch_size[0], 3, args.height[0], args.width[0]],
        #                 max_shape=[args.batch_size[0], 3, args.height[0], args.width[0]])))
        # else:
        #     assert len(args.batch_size) == len(deploy_cfg['onnx_config']['output_names']), ValueError(f"[ERROR] the number of batch-size({len(args.batch_size)}) must be same to the number of output names({len(deploy_cfg['onnx_config']['output_names'])})")
        #     deploy_cfg['backend_config']['model_inputs'][idx] = dict(
        #         input_shapes=dict(
        #             input=dict(
        #                 min_shape=[args.batch_size[idx], 3, args.height[idx], args.width[idx]],
        #                 opt_shape=[args.batch_size[idx], 3, args.height[idx], args.width[idx]],
        #                 max_shape=[args.batch_size[idx], 3, args.height[idx], args.width[idx]])))
        
    save_file = get_ir_config(deploy_cfg)['save_file']

    import torch
    model_inputs = torch.zeros(args.batch_size, 3, args.height, args.width)

    torch2onnx(
        model_inputs,
        args.work_dir,
        save_file,
        deploy_cfg=deploy_cfg,
        model_cfg=args.model_cfg,
        model_checkpoint=args.checkpoint,
        device=args.device)


if __name__ == '__main__':
    main()
