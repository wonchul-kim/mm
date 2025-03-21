
import os.path as osp
import warnings

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

class BaseConfigManager:
    _cls = None 
    
    def __init__(self, cfg=None):
        self._cfg = cfg 
        self._args = None
        self.manage_model_config = None
        
    @property
    def cfg(self):
        return self._cfg 
            
    @property
    def args(self):
        return self._args 
    
    def build(self, args, config_file):
                
        # if 'dataset_type' in args and args.dataset_type in ['mask', 'labelme']:
        #     create_custom_dataset(args.dataset_type)
            
        self._cfg, self._args = self.build_config(args, config_file)
        
        if args.model == 'yolov8':
            self.manage_model_config = self.manage_yolov8_config
        else:
            raise NotImplementedError(f"{args.model} is NOT Considered")

        if hasattr(args, 'frozen_stages'):
            self.manage_frozen_stages(args.frozen_stages)
    
    def manage_schedule_config(self, max_epochs, val_interval):
        self._cfg.max_epochs = max_epochs 
        self._cfg.default_hooks.param_scheduler.max_epochs = max_epochs 
        
        for custom_hook in self._cfg.custom_hooks:
            if custom_hook.get('type') == 'mmdet.PipelineSwitchHook':
                custom_hook.switch_epoch = max_epochs - self._cfg.close_mosaic_epochs
        
        def _manage_train_loop(cfg):
            if cfg.train_cfg.get('type') == 'EpochBasedTrainLoop':
                cfg.train_cfg.max_epochs = max_epochs
                cfg.train_cfg.val_interval = val_interval
                cfg.train_cfg.dynamic_intervals=[((max_epochs - cfg.close_mosaic_epochs), cfg.val_interval_stage2)]
                
        _manage_train_loop(self._cfg)
                   
    
    def manage_dataset_config(self, data_root, img_suffix, seg_map_suffix, 
                                    classes, batch_size, width, height, 
                                    rois, patch):
        img_scale = (width, height)
        for pipeline in self._cfg.train_pipeline:
            if 'img_scale' in pipeline.keys():
                pipeline['img_scale'] = img_scale 
            
            if pipeline.get('type') == 'YOLOv5RandomAffine':
                pipeline.border = (-img_scale[0] // 2, -img_scale[1] // 2)
            
        for pipeline in self._cfg.train_pipeline_stage2:
            if 'img_scale' in pipeline.keys():
                pipeline['img_scale'] = img_scale 
            
            if pipeline.get('type') in ['YOLOv5KeepRatioResize', 'LetterResize']:
                pipeline.scale = img_scale
                
        for pipeline in self._cfg.test_pipeline:
            if 'img_scale' in pipeline.keys():
                pipeline['img_scale'] = img_scale 
            
            if pipeline.get('type') in ['YOLOv5KeepRatioResize', 'LetterResize']:
                pipeline.scale = img_scale
            
                
        self._cfg.train_dataloader.dataset.pipeline = self._cfg.train_pipeline
        self._cfg.val_dataloader.dataset.pipeline = self._cfg.test_pipeline
        self._cfg.test_dataloader = self._cfg.val_dataloader
        
        for custom_hook in self._cfg.custom_hooks:
            if custom_hook.get('type') == 'mmdet.PipelineSwitchHook':
                custom_hook.switch_pipeline = self._cfg.train_pipeline_stage2
            
            
                   
    # # set dataset ====================================================================================================
    # def manage_dataset_config(self, data_root, img_suffix, seg_map_suffix, 
    #                                 classes, batch_size, width, height, 
    #                                 rois, patch):
    #     def _manage_train_dataloader(cfg):
    #         cfg.train_dataloader.batch_size = batch_size
    #         cfg.train_dataloader.dataset['data_root'] = data_root
    #         cfg.train_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
    #         cfg.train_dataloader.dataset['classes'] = classes
    #         cfg.train_dataloader.dataset['img_suffix'] = img_suffix
    #         cfg.train_dataloader.dataset['rois'] = rois
    #         cfg.train_dataloader.dataset['patch'] = patch
            
    #     def _manage_val_dataloader(cfg):
    #         cfg.val_dataloader.batch_size = batch_size
    #         cfg.val_dataloader.dataset['data_root'] = data_root
    #         cfg.val_dataloader.dataset['classes'] = classes
    #         cfg.val_dataloader.dataset['img_suffix'] = img_suffix
    #         cfg.val_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
    #         cfg.val_dataloader.dataset['rois'] = rois
    #         cfg.val_dataloader.dataset['patch'] = patch
            
    #     def _manage_test_dataloader(cfg):
    #         cfg.test_dataloader.batch_size = batch_size
    #         cfg.test_dataloader.dataset['data_root'] = data_root
    #         cfg.test_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
    #         cfg.test_dataloader.dataset['classes'] = classes
    #         cfg.test_dataloader.dataset['img_suffix'] = img_suffix
    #         cfg.test_dataloader.dataset['rois'] = rois
    #         cfg.test_dataloader.dataset['patch'] = patch
        
    #     def _manage_crop_size(cfg, width, height, patch=None):
    #         if 'train_pipeline' in cfg and isinstance(cfg.train_pipeline, list):
    #             for pipeline in cfg.train_pipeline:
    #                 if pipeline.get('type') == 'Resize':
    #                     pipeline['scale'] = (width, height)
                        
    #                 if pipeline.get('type') == 'RandomResize':
    #                     pipeline['scale'] = (width, height)
                        
    #                 if pipeline.get('type') == 'RandomCrop':
    #                     pipeline['crop_size'] = (height, width)
                    
    #             if cfg.dataset_type == 'LabelmeDataset' and not any(step.get('type') in ['LoadAnnotations', 'LoadLabelmeAnnotations'] for step in cfg.train_pipeline):
    #                 cfg.train_pipeline.insert(1, dict(type='LoadLabelmeAnnotations', reduce_zero_label=False))
    #             else:
    #                 cfg.train_pipeline.insert(1, dict(type='LoadAnnotations', reduce_zero_label=False))

    #         if 'val_pipeline' in cfg and isinstance(cfg.val_pipeline, list):
    #             for pipeline in cfg.val_pipeline:
    #                 if pipeline.get('type') == 'Resize':
    #                     pipeline['scale'] = (width, height)
                    
    #             if cfg.dataset_type == 'LabelmeDataset' and not any(step.get('type') in ['LoadAnnotations', 'LoadLabelmeAnnotations'] for step in cfg.val_pipeline):
    #                 cfg.val_pipeline.insert(2, dict(type='LoadLabelmeAnnotations', reduce_zero_label=False))
    #             else:
    #                 cfg.val_pipeline.insert(2, dict(type='LoadAnnotations', reduce_zero_label=False))
                    
    #         if 'test_pipeline' in cfg and isinstance(cfg.test_pipeline, list):
    #             resize_index = 0
    #             for idx, pipeline in enumerate(cfg.test_pipeline):
    #                 if pipeline.get('type') == 'Resize':
    #                     pipeline['scale'] = (width, height)
    #                     resize_index = idx
                        
    #             # if patch and patch['use_patch']:
    #             #     del cfg.test_pipeline[resize_index]
                        
    #             if cfg.dataset_type == 'LabelmeDataset' and not any(step.get('type') in ['LoadAnnotations', 'LoadLabelmeAnnotations'] for step in cfg.test_pipeline):
    #                 cfg.test_pipeline.insert(2, dict(type='LoadLabelmeAnnotations', reduce_zero_label=False))
    #             else:
    #                 cfg.test_pipeline.insert(2, dict(type='LoadAnnotations', reduce_zero_label=False))
                    
    #         if hasattr(self._cfg, 'train_dataloader'):
    #             cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    #         if hasattr(self._cfg, 'val_dataloader'):
    #             cfg.val_dataloader.dataset.pipeline = cfg.val_pipeline
    #         if hasattr(self._cfg, 'test_dataloader'):
    #             cfg.test_dataloader.dataset.pipeline = cfg.test_pipeline

    #     if hasattr(self._cfg, 'train_dataloader'):
    #         _manage_train_dataloader(self._cfg)
    #     if hasattr(self._cfg, 'val_dataloader'):
    #         _manage_val_dataloader(self._cfg)            
    #     if hasattr(self._cfg, 'test_dataloader'):
    #         _manage_test_dataloader(self._cfg)
            
    #     if hasattr(self._cfg, 'height'):
    #         self._cfg.height = height
    #     if hasattr(self._cfg, 'width'):
    #         self._cfg.width = width

    #     _manage_crop_size(self._cfg, width, height, self.args.patch if hasattr(self.args, 'patch') else None)

    # set model configs ==============================================================================
    def manage_yolov8_config(self, num_classes, width, height):
        
        def _manage_num_classes(cfg):
            cfg.num_classes = num_classes 
            if 'model' in cfg:
                if cfg.model.get('type') == 'YOLODetector':
                    cfg.model.bbox_head.head_module.num_classes = num_classes
                    cfg.model.train_cfg.assigner.num_classes = num_classes 
                        
        def _manage_img_scale(cfg, new_img_scale): # img_scale: (width, height)
            cfg.img_scale = new_img_scale 
            # cfg.data_preprocessor.size = new_crop_size
            # cfg.model.data_preprocessor = cfg.data_preprocessor

        _manage_num_classes(self._cfg)
        _manage_img_scale(self._cfg, (width, height))
        
        
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

    # set defulat_hooks ===============================================================================
    def manage_default_hooks_config(self, default_hooks):
        pass

    # set custom_hooks ================================================================================
    def manage_custom_hooks_config(self, custom_hooks):
        _custom_hooks = []
        for key, val in custom_hooks.items():
            if key == 'checkpoint':
                _custom_hooks.append(dict(type='CustomCheckpointHook', interval=val.get('interval', 100),
                                        by_epoch=val.get('by_epoch', False), save_best=val.get('save_best', 'mIoU'),
                                        max_keep_ckpts=val.get('max_keep_ckpts', 2),
                                        out_dir=val.get('output_dir', osp.join(self._cfg.work_dir, 'weights')))
                                    )
            
            elif key == 'visualize_val':
                if 'output_dir' not in val.keys() or val['output_dir'] == None:
                    output_dir = osp.join(self._cfg.work_dir, 'val')
                else:
                    output_dir = val['output_dir']
                _custom_hooks.append(dict(type='VisualizeVal', freq_epoch=val.get('freq_epoch', 1), 
                                                   ratio=val.get('ratio', 0.25), 
                                                   output_dir=output_dir))
            
            elif key == 'before_train':
                for key2, val2 in val.items():
                    if key2 == 'debug_dataloader':
                        _custom_hooks.append(dict(type='HookBeforeTrain', ratio=val2.get('ratio', 0.25),
                                                debug_dir=val2.get('output_dir', osp.join(self._cfg.work_dir, 'debug_dir'))))

            elif key == 'after_train_epoch':
                _custom_hooks.append(dict(type='HookAfterTrainIter'))
            elif key == 'after_val_epoch':
                _custom_hooks.append(dict(type='HookAfterValEpoch'))
                
            elif key == 'aiv':
                if val.get('use', False):
                    aiv = True
                    for key2, val2 in val.items():
                        if key2 == 'logging':
                            logs_dir = val2.get('output_dir', osp.join(self._cfg.work_dir, 'logs_dir'))
                            
                            for key3, val3 in val2.items():
                                if key3 == 'monitor' and val3.get('use', False):
                                    _custom_hooks.append(dict(type='HookForAiv', aiv=aiv,
                                            monitor=True,
                                            monitor_csv=val3.get('monitor_csv', False), monitor_figs=val3.get('monitor_figs', False),
                                            monitor_freq=val3.get('monitor_freq', 1), logs_dir=logs_dir))
                
        
        if len(_custom_hooks) != 0:
            if not hasattr(self._cfg, 'custom_hooks'):
                self._cfg.custom_hooks = _custom_hooks
            else:
                self._cfg.custom_hooks += _custom_hooks
                
    # set custom_hooks ================================================================================
    def manage_custom_test_hooks_config(self, custom_hooks):
        _custom_hooks = []
        for key, val in custom_hooks.items():
            if key == 'visualize_test':
                if 'output_dir' not in val.keys() or val['output_dir'] == None:
                    raise RuntimeError(f"Output directory must be defined")
                else:
                    output_dir = val['output_dir']
                    
                _custom_hooks.append(dict(type='VisualizeTest', output_dir=output_dir, 
                                          annotate=val.get('annotate', False), 
                                          contour_thres=val.get('contour_thres', 10),
                                          contour_conf=val.get('contour_conf', 0.5)))
            
            elif key == 'aiv':
                if val.get('use', False):
                    aiv = True
                    for key2, val2 in val.items():
                        if key2 == 'logging':
                            logs_dir = val2.get('output_dir', osp.join(self._cfg.work_dir, 'logs_dir'))
                            
                            for key3, val3 in val2.items():
                                if key3 == 'monitor' and val3.get('use', False):
                                    _custom_hooks.append(dict(type='HookForAiv', aiv=aiv,
                                            monitor=True,
                                            monitor_csv=val3.get('monitor_csv', False), monitor_figs=val3.get('monitor_figs', False),
                                            monitor_freq=val3.get('monitor_freq', 1), logs_dir=logs_dir))
                
        
        if len(_custom_hooks) != 0:
            if not hasattr(self._cfg, 'custom_hooks'):
                self._cfg.custom_hooks = _custom_hooks
            else:
                self._cfg.custom_hooks += _custom_hooks
        
    # freeze backbone ==================================================================================
    def manage_frozen_stages(self, frozen_stages):
        
        if frozen_stages != -1:

            if hasattr(self._cfg.model.backbone, 'frozen_stages'):
                if self._cfg.model.backbone.type == 'SwinTransformer':
                    assert frozen_stages >= 0 and frozen_stages <= 4, ValueError(f'The `frozen_stages` must be 0 <= frozen_stages <= 4, not {frozen_stages}')
                
                elif self._cfg.model.backbone.type == 'COSNet':
                    assert frozen_stages >= 0 and frozen_stages <= 3, ValueError(f'The `frozen_stages` must be 0 <= frozen_stages <= 3, not {frozen_stages}')
                elif self._cfg.model.backbone.type == 'ResNetV1c':
                    assert frozen_stages >= 0 and frozen_stages <= 4, ValueError(f'The `frozen_stages` must be 0 <= frozen_stages <= 4, not {frozen_stages}')
                elif self._cfg.model.backbone.type == 'PIDNet':
                    assert frozen_stages >= 0 and frozen_stages <= 4, ValueError(f'The `frozen_stages` must be 0 <= frozen_stages <= 4, not {frozen_stages}')
                elif self._cfg.model.backbone.type == 'DinoVisionTransformer':
                    assert frozen_stages >= 0 and frozen_stages <= 24, ValueError(f'The `frozen_stages` must be 0 <= frozen_stages <= 24, not {frozen_stages}')
                elif self._cfg.model.backbone.type == 'HetNetEncoder':
                    assert frozen_stages >= 0 and frozen_stages <= 4, ValueError(f'The `frozen_stages` must be 0 <= frozen_stages <= 4, not {frozen_stages}')
                else:
                    warnings.warn(f"There is not yet `frozen_stages` considered in backbone({self._cfg.model.backbone.type})")
                
                self._cfg.model.backbone.frozen_stages = frozen_stages
            else:
                raise RuntimeError(f"There is no `frozen_stages` in backbone, YOU NEED TO CHECK!")
            
        
                
    