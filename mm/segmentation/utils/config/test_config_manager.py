from mmengine.config import Config
from .base_config_manager import BaseConfigManager, create_custom_dataset

class TestConfigManager(BaseConfigManager):
    def __init__(self, cfg=None):
        super().__init__(cfg=cfg)
        
    @classmethod
    def build_config(cls, args, config_file):
                
        if args.cfg_options is None:
            cfg_options = {'load_from': args.load_from,
                    'launcher': args.launcher, 
                    # 'resume': args.resume,
                    'work_dir': args.output_dir,
                    # 'show_dir': args.show_dir, 
                    'tta': args.tta,
                    # 'wait_time': 2,
                    # 'out': args.out, 
                    # 'show': args.show,
                }
        else:
            cfg_options = args.cfg_options
            
        cfg = Config.fromfile(config_file)
        cfg.merge_from_dict(cfg_options)
    
        return cfg, args
                

