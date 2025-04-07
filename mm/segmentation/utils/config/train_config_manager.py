
import os.path as osp
from mmengine.config import Config
from mmengine.logging import print_log
import logging
from .base_config_manager import BaseConfigManager, create_custom_dataset

class TrainConfigManager(BaseConfigManager):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        
    @classmethod
    def build_config(cls, args, config_file):
        
        if args.cfg_options is None:
            cfg_options = {'load_from': args.load_from,
                    'launcher': args.launcher, 
                    'resume': args.resume,
                    'work_dir': args.output_dir
                }
        else:
            cfg_options = args.cfg_options
            
        cfg = Config.fromfile(config_file)
        cfg.merge_from_dict(cfg_options)
    
        if args.amp is True and hasattr(cfg, 'optim_wrapper'):
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
                
        return cfg, args
                
