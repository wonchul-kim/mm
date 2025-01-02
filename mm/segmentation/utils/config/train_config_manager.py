
from mmengine.config import Config
import os.path as osp

def create_custom_dataset(dataset_type):
    import importlib.util
    import os
    
    module_name = "mm"
    spec = importlib.util.find_spec(module_name)

    if spec is not None:
        module_path = spec.origin
        package_path = os.path.dirname(module_path)
        print(f"The 'mm' package is installed at: {package_path}")
    else:
        print(f"The 'mm' package is not installed.")
        
    dataset_path = os.path.join(package_path, f'segmentation/configs/_base_/datasets/{dataset_type}.py')
        
    content = f"_base_ = ['{dataset_path}']"
    with open('/tmp/custom_dataset.py', "w") as file:
        file.write(content)

class TrainConfigManager:
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
        
        if 'dataset_type' in args and args.dataset_type in ['mask', 'labelme']:
            create_custom_dataset(args.dataset_type)
        
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
        
        def _manage_crop_size(cfg, new_crop_size):
            if 'train_pipeline' in cfg and isinstance(cfg.train_pipeline, list):
                for pipeline in cfg.train_pipeline:
                    if pipeline.get('type') == 'RandomCrop':
                        pipeline['crop_size'] = tuple(new_crop_size)
                        
                    
                if cfg.dataset_type == 'LabelmeDataset' and not any(step.get('type') in ['LoadAnnotations', 'LoadLabelmeAnnotations'] for step in cfg.train_pipeline):
                    cfg.train_pipeline.insert(1, dict(type='LoadLabelmeAnnotations', reduce_zero_label=True))
                else:
                    cfg.train_pipeline.insert(1, dict(type='LoadAnnotations', reduce_zero_label=True))
                        
            if 'test_pipeline' in cfg and isinstance(cfg.test_pipeline, list):
                for pipeline in cfg.test_pipeline:
                    if pipeline.get('type') == 'Resize':
                        pipeline['scale'] = tuple(new_crop_size)
                        
                if cfg.dataset_type == 'LabelmeDataset' and not any(step.get('type') in ['LoadAnnotations', 'LoadLabelmeAnnotations'] for step in cfg.test_pipeline):
                    cfg.test_pipeline.insert(1, dict(type='LoadLabelmeAnnotations', reduce_zero_label=True))
                else:
                    cfg.test_pipeline.insert(1, dict(type='LoadAnnotations', reduce_zero_label=True))
                    
            cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
            cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline
            
        _manage_train_dataloader(self._cfg)
        _manage_val_dataloader(self._cfg)
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
        
        
    # set dataloader ==================================================================================
    def manage_dataloader_config(self, vis_dataloader_ratio):
        def _manage_train_dataloader(cfg):
            cfg.train_dataloader.vis_dataloader_ratio = vis_dataloader_ratio
            cfg.train_dataloader.vis_dir = osp.join(cfg.work_dir, 'dataloader')
        
        def _manage_val_dataloader(cfg):
            cfg.val_dataloader.vis_dataloader_ratio = vis_dataloader_ratio
            cfg.val_dataloader.vis_dir = osp.join(cfg.work_dir, 'dataloader')
        
        _manage_train_dataloader(self._cfg)
        _manage_val_dataloader(self._cfg)
