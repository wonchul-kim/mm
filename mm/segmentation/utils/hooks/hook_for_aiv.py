import numpy as np
import os
import os.path as osp
from typing import Optional, Sequence, Union

from mmseg.registry import HOOKS
from mmengine.hooks import Hook

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