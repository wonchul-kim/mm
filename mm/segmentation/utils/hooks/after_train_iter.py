

from typing import Optional, Sequence, Union
import numpy as np
from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.runner.loops import _parse_losses, _update_losses

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookAfterTrainIter(Hook):
    def __init__(self, **kwargs):
        pass        

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         outputs: Optional[dict] = None) -> None:     
        
        pass
        
                
            
            
            
        