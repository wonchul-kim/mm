import numpy as np
from typing import Optional, Sequence, Union

from mmseg.registry import HOOKS
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookAfterValEpoch(Hook):
    def __init__(self, **kwargs):
        pass        

    def after_val_epoch(self, runner, metrics) -> None:

        classes = metrics['Class']
        val_log = {}
        for key, val in metrics.items():
            if isinstance(val, (float, int)):
                val_log[key] = val
            elif isinstance(val, (np.ndarray, list)):
                for _val, _class in zip(val, classes):
                    val_log[f'{key}_{_class}'] = _val
                    
        runner.val_log = val_log
        if hasattr(runner, 'aiv_val_monitor'):
            runner.aiv_val_monitor.log(val_log)
            runner.aiv_val_monitor.save()