from mmengine.config import Config


class TestConfigManager:
    _cls = None 
    
    def __init__(self, cfg=None):
        self._cfg = cfg 
        
    @property
    def cfg(self):
        return self._cfg 
            
    def build(self, args, config_file):
        self._cfg = self.build_config(args, config_file)
    
    @classmethod
    def build_config(cls, args, config_file):
        if args.cfg_options is None:
            cfg_options = {'load_from': args.load_from,
                    'launcher': args.launcher, 
                    # 'resume': args.resume,
                    'work_dir': args.output_dir,
                    'show_dir': args.show_dir, 
                    'tta': args.tta,
                    'wait_time': 2,
                    'out': args.out, 
                    'show': args.show,
                }
        else:
            cfg_options = args.cfg_options
            
        cfg = Config.fromfile(config_file)
        cfg.merge_from_dict(cfg_options)
    
        if args.amp is True:
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
                
        return cfg
                
    def manage_schedule_config(self, max_iters, val_interval, checkpoint_interval):
        def _manage_train_loop(cfg):
            if cfg.train_cfg.get('type') == 'IterBasedTrainLoop':
                cfg.train_cfg.max_iters = max_iters
                cfg.train_cfg.val_interval = val_interval
        def _manage_param_scheduler(cfg):
            if 'param_scheduler' in cfg and isinstance(cfg.param_scheduler, list):
                for scheduler in cfg.param_scheduler:
                    if scheduler.get('type') == 'PolyLR':
                        scheduler['end'] = max_iters
                        
        def _manage_checkpoint_interval(cfg):
            if 'checkpoint' in cfg.default_hooks:
                cfg.default_hooks.checkpoint.interval = checkpoint_interval
                
        _manage_train_loop(self._cfg)
        _manage_param_scheduler(self._cfg)
        _manage_checkpoint_interval(self._cfg)
            
    # set dataset ====================================================================================================
    def manage_dataset_config(self, data_root, img_suffix, seg_map_suffix, classes, batch_size, width, height):
        def _manage_test_dataloader(cfg):
            cfg.test_dataloader.batch_size = batch_size
            cfg.test_dataloader.dataset['data_root'] = data_root
            cfg.test_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
            cfg.test_dataloader.dataset['classes'] = classes
            cfg.test_dataloader.dataset['img_suffix'] = img_suffix
        
        def _manage_crop_size(cfg, new_crop_size):
            if 'test_pipeline' in cfg and isinstance(cfg.test_pipeline, list):
                for pipeline in cfg.test_pipeline:
                    if pipeline.get('type') == 'Resize':
                        pipeline['scale'] = tuple(new_crop_size)
                    
            cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline
            
        _manage_test_dataloader(self._cfg)
        _manage_crop_size(self._cfg, (width, height))
        
    # set num_classes =================================================================================
    def manage_model_config(self, num_classes, width, height):
        
        def _manage_num_classes(cfg):
            cfg.num_classes = num_classes 
            if 'model' in cfg:
                if cfg.model.get('type') == 'EncoderDecoder':
                    if 'decode_head' in cfg.model and 'num_classes' in cfg.model.decode_head:
                        cfg.model.decode_head.num_classes = num_classes
                        cfg.model.decode_head.loss_cls.class_weight = [1.0] * num_classes + [0.1]
                        
        def _manage_crop_size(cfg, new_crop_size):
            cfg.crop_size = new_crop_size 
            cfg.data_preprocessor.size = new_crop_size
            cfg.model.data_preprocessor = cfg.data_preprocessor

        _manage_num_classes(self._cfg)
        _manage_crop_size(self._cfg, (height, width))
        
