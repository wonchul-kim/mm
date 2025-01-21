

from typing import Optional, Sequence, Union

from mmseg.registry import HOOKS
from mmengine.hooks import Hook

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
        
        if batch_idx != 0 and batch_idx%len(runner.train_dataloader) == 0:
            train_log = {}
            if 'loss' in outputs:
                train_log['loss'] = outputs['loss'].item()
                
            for idx, base_lr in enumerate(runner.optim_wrapper.get_lr()['base_lr']):
                train_log[f'lr_{idx}'] = base_lr
            
            runner.train_log = train_log
            
        # TODO: metrics for train dataset
            
            
            
            
        