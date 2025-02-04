import torch
import os.path as osp
from .base_config_manager import BaseConfigManager
from mmdeploy.utils import load_config

class ExportConfigManager(BaseConfigManager):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        
    def build(self, args, config_file):
                
        # args.batch_size = to_list(args.batch_size)
        # args.width = to_list(args.width)
        # args.height = to_list(args.height)
        
        # assert len(args.batch_size) == len(args.height), ValueError(f"[ERROR] the number of batch-size({len(args.batch_size)}) must be same to the number of height({len(args.height)})")
        # assert len(args.batch_size) == len(args.width), ValueError(f"[ERROR] the number of batch-size({len(args.batch_size)}) must be same to the number of width({len(args.width)})")
                
        self._cfg = load_config(config_file)[0]
        
        self._cfg['model_inputs'] = torch.zeros(args.batch_size, 3, args.height, args.width)
        
        for key, val in args.codebase_config.items():
            self._cfg['codebase_config'][key] = val
        
        for key, val in args.onnx_config.items():
            self._cfg['onnx_config'][key] = val
            
        self._cfg['onnx_config']['input_shape'] = [args.width, args.height]
        self._cfg['onnx_config']['save_file'] = osp.join(args.work_dir, f'{args.model}_{args.backbone}_b{args.batch_size}_w{args.width}_h{args.height}')
            
        for key, val in args.backend_config.items():
            self._cfg['backend_config'][key] = val
            
        for idx in range(len(self._cfg['onnx_config']['output_names'])):
            self._cfg['backend_config']['model_inputs'][idx] = dict(
                    input_shapes=dict(
                        input=dict(
                            min_shape=[args.batch_size, 3, args.height, args.width],
                            opt_shape=[args.batch_size, 3, args.height, args.width],
                            max_shape=[args.batch_size, 3, args.height, args.width])))
            # if len(args.batch_size) == 1:
            #     self._cfg['backend_config']['model_inputs'][idx] = dict(
            #         input_shapes=dict(
            #             input=dict(
            #                 min_shape=[args.batch_size[0], 3, args.height[0], args.width[0]],
            #                 opt_shape=[args.batch_size[0], 3, args.height[0], args.width[0]],
            #                 max_shape=[args.batch_size[0], 3, args.height[0], args.width[0]])))
            # else:
            #     assert len(args.batch_size) == len(self._cfg['onnx_config']['output_names']), ValueError(f"[ERROR] the number of batch-size({len(args.batch_size)}) must be same to the number of output names({len(self._cfg['onnx_config']['output_names'])})")
            #     self._cfg['backend_config']['model_inputs'][idx] = dict(
            #         input_shapes=dict(
            #             input=dict(
            #                 min_shape=[args.batch_size[idx], 3, args.height[idx], args.width[idx]],
            #                 opt_shape=[args.batch_size[idx], 3, args.height[idx], args.width[idx]],
            #                 max_shape=[args.batch_size[idx], 3, args.height[idx], args.width[idx]])))

