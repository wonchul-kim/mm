import numpy as np
import os
from typing import Optional, Sequence, Union, Dict
import time 

from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.runner.loops import _parse_losses

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookForAiv(Hook):
    def __init__(self, aiv, **kwargs):
        self.aiv = aiv
        self.monitor = kwargs['monitor']
        self.monitor_csv = kwargs['monitor_csv']
        self.monitor_figs = kwargs['monitor_figs']
        self.monitor_freq = kwargs['monitor_freq']
        self.logs_dir = kwargs['logs_dir']
                
        
    def before_run(self, runner) -> None:
        ## set custom logger, monitor
        if self.aiv:
            # set logs =================================================================
            runner_attributes = ['train_log', 'val_log']
            for runner_attribute in runner_attributes:
                setattr(runner, runner_attribute, None)
            
            
            # set monitor =================================================================
            from aivcommon.loggings import Monitor
            import os.path as osp
            
            setattr(runner, 'aiv_train_monitor', Monitor())
            setattr(runner, 'aiv_val_monitor', Monitor())
            
            if not osp.exists(osp.join(self.logs_dir, "train")):
                os.makedirs(osp.join(self.logs_dir, "train"), exist_ok=True)
            if not osp.exists(osp.join(self.logs_dir, "val")):
                os.makedirs(osp.join(self.logs_dir, "val"), exist_ok=True)
                
            runner.aiv_train_monitor.set(
                output_dir=osp.join(self.logs_dir, "train"),
                fn="train",
                use=self.monitor,
                save_csv=self.monitor_csv,
                save_figs=self.monitor_figs,
                save_freq=self.monitor_freq,
            )
            runner.aiv_val_monitor.set(
                output_dir=osp.join(self.logs_dir, "val"),
                fn="val",
                use=self.monitor,
                save_csv=self.monitor_csv,
                save_figs=self.monitor_figs,
                save_freq=self.monitor_freq,
            )
            
    def before_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        
        ## system log =================================================================
        try:
            import torch
            from visionsuite.engines.utils.system import GPULogger, CPULogger 
            
            def get_gpus():
                if torch.cuda.is_available():
                    return list(range(torch.cuda.device_count()))
                else:
                    return []
            
            gpu_list = get_gpus()
            if len(gpu_list) != 0:
                runner.train_gpu_logger = GPULogger(gpu_list)
                runner.val_gpu_logger = GPULogger(gpu_list)
                
            # runner.train_cpu_logger = CPULogger()
            # runner.val_cpu_logger = CPULogger()
        except:
            pass
        
        ## time duration log ========================================================
        runner.time_duration = {'train': {'tic': time.perf_counter(), 'tac': -1},
                                'val': {'tic': -1, 'tac': -1}}
            
    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:     
                           
        ## evaluate ===================================================================
        metrics = {}
        if hasattr(runner.train_loop, 'evaluator') and runner.train_loop.evaluator:
            runner.train_loop.run_eval_iter(data_batch)

            if batch_idx != 0 and batch_idx%len(runner.train_dataloader) == 0:
                metrics = runner.train_loop.evaluator.evaluate(len(runner.train_dataloader.dataset))

                if runner.train_loop.train_loss:
                    loss_dict = _parse_losses(runner.train_loop.train_loss, 'train')
                    metrics.update(loss_dict)
                    runner.train_loop.train_loss.clear()
        
        ## system log ==================================================================
        if hasattr(runner, 'train_gpu_logger'):
            runner.train_gpu_logger.update()
        # if hasattr(runner, 'train_cpu_logger'):
        #     runner.train_cpu_logger.update()
        
        ## hyper-parameter log =======================================================================
        if batch_idx != 0 and batch_idx%len(runner.train_dataloader) == 0:
            train_log = {}
            # loss
            if 'loss' in outputs:
                train_log['loss'] = outputs['loss'].item()
                
            # lr
            lr_dict = runner.optim_wrapper.get_lr()
            if 'base_lr' in lr_dict:
                lr_list = lr_dict['base_lr']
            elif 'lr' in lr_dict:
                lr_list = lr_dict['lr']        
            else:
                raise NotImplementedError(f"[ERROR] NOT Considered this case of lr: {lr_dict}")
              
            for idx, base_lr in enumerate(lr_list):
                train_log[f'lr_{idx}'] = base_lr
            
            # system
            if hasattr(runner, 'train_gpu_logger'):
                for gpu, gpu_log in runner.train_gpu_logger.mean().items():
                    for key, val in gpu_log.items():
                        train_log[f"{gpu} {key}"] = val
                runner.train_gpu_logger.clear()
                    
            # time
            if hasattr(runner, 'time_duration'):
                runner.time_duration['train']['tac'] = time.perf_counter()
                train_log['duration (sec)'] = runner.time_duration['train']['tac'] - runner.time_duration['train']['tic']
                runner.time_duration['train']['tic'] = time.perf_counter()
                
            # save --------------------------------------------------------
            runner.train_log = train_log
            if hasattr(runner, 'aiv_train_monitor'):
                
                if len(metrics) != 0:
                    classes = metrics['Class']
                    for key, val in metrics.items():
                        if isinstance(val, (float, int)):
                            train_log[key] = val
                        elif isinstance(val, (np.ndarray, list)):
                            for _val, _class in zip(val, classes):
                                train_log[f'{key}_{_class}'] = _val
                
                runner.aiv_train_monitor.log(train_log)
                runner.aiv_train_monitor.save()
                
            
    def after_train_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations after each training epoch.

        Args:
            runner (Runner): The runner of the training process.
        """
        if hasattr(runner, 'train_gpu_logger'):
            runner.train_gpu_logger.clear()
        # if hasattr(runner, 'train_cpu_logger'):
        #     runner.train_cpu_logger.clear()
                    
    def before_val_epoch(self, runner) -> None:
        """All subclasses should override this method, if they need any
        operations before each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
        """
        # time
        runner.time_duration['val']['tic'] = time.perf_counter()

    def after_val_iter(self,
                    runner,
                    batch_idx: int,
                    data_batch: DATA_BATCH = None,
                    outputs: Optional[Sequence] = None) -> None:
        
        if hasattr(runner, 'val_gpu_logger'):
            runner.val_gpu_logger.update()
        # if hasattr(runner, 'val_cpu_logger'):
        #     runner.val_cpu_logger.update()
            
    def after_val_epoch(self,
                        runner,
                        metrics: Optional[Dict[str, float]] = None) -> None:
        """All subclasses should override this method, if they need any
        operations after each validation epoch.

        Args:
            runner (Runner): The runner of the validation process.
            metrics (Dict[str, float], optional): Evaluation results of all
                metrics on validation dataset. The keys are the names of the
                metrics, and the values are corresponding results.
        """

        classes = metrics['Class']
        val_log = {}
        for key, val in metrics.items():
            if isinstance(val, (float, int)):
                val_log[key] = val
            elif isinstance(val, (np.ndarray, list)):
                for _val, _class in zip(val, classes):
                    val_log[f'{key}_{_class}'] = _val
                
        # system
        if hasattr(runner, 'val_gpu_logger'):
            for gpu, gpu_log in runner.val_gpu_logger.mean().items():
                for key, val in gpu_log.items():
                    val_log[f"{gpu} {key}"] = val
                    
            runner.val_gpu_logger.clear()
                    
        # if hasattr(runner, 'val_cpu_logger'):
        #     runner.val_cpu_logger.clear()
            
        # time
            if hasattr(runner, 'time_duration'):
                runner.time_duration['val']['tac'] = time.perf_counter()
                val_log['duration (sec)'] = runner.time_duration['val']['tac'] - runner.time_duration['val']['tic']
                
        # save --------------------------------------------
        runner.val_log = val_log
        if hasattr(runner, 'aiv_val_monitor'):
            runner.aiv_val_monitor.log(val_log)
            runner.aiv_val_monitor.save()