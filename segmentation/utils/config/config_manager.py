

class ConfigManager:
    def __init__(self, cfg):
        self._cfg = cfg 
        
    @property
    def cfg(self):
        return self._cfg 
    
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
    def manage_dataset_config(self, data_root, img_suffix, seg_map_suffix, classes, batch_size, new_crop_size):
        def _manage_train_dataloader(cfg):
            cfg.train_dataloader.batch_size = batch_size
            cfg.train_dataloader.dataset['data_root'] = data_root
            cfg.train_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
            cfg.train_dataloader.dataset['classes'] = classes
            cfg.train_dataloader.dataset['img_suffix'] = img_suffix
        
        def _manage_val_dataloader(cfg):
            cfg.val_dataloader.batch_size = batch_size
            cfg.val_dataloader.dataset['data_root'] = data_root
            cfg.val_dataloader.dataset['classes'] = classes
            cfg.val_dataloader.dataset['img_suffix'] = img_suffix
            cfg.val_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
        
        def _manage_crop_size(cfg):
            if 'train_pipeline' in cfg and isinstance(cfg.train_pipeline, list):
                for pipeline in cfg.train_pipeline:
                    if pipeline.get('type') == 'RandomCrop':
                        pipeline['crop_size'] = tuple(new_crop_size)
                        
                        
            if 'test_pipeline' in cfg and isinstance(cfg.test_pipeline, list):
                for pipeline in cfg.test_pipeline:
                    if pipeline.get('type') == 'RandomCrop':
                        pipeline['crop_size'] = tuple(new_crop_size)
                    
            cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
            cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
            
        _manage_train_dataloader(self._cfg)
        _manage_val_dataloader(self._cfg)
        _manage_crop_size(self._cfg)
        
    # set num_classes =================================================================================
    def manage_model_config(self, num_classes, new_crop_size):
        def _manage_num_classes(cfg):
            cfg.num_classes = num_classes 
            if 'model' in cfg:
                if cfg.model.get('type') == 'EncoderDecoder':
                    if 'decode_head' in cfg.model and 'num_classes' in cfg.model.decode_head:
                        cfg.model.decode_head.num_classes = num_classes
                        cfg.model.decode_head.loss_cls.class_weight = [1.0] * num_classes + [0.1]
                        
        def _manage_crop_size(cfg):
            cfg.crop_size = new_crop_size 
            cfg.data_preprocessor.size = new_crop_size
            cfg.model.data_preprocessor = cfg.data_preprocessor

        _manage_num_classes(self._cfg)
        _manage_crop_size(self._cfg)