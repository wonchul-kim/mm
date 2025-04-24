

from typing import Optional, Union
from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from mm.segmentation.utils.class_weights import change_class_weights_to_loss_decode

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookBeforeTrainIter(Hook):
    def __init__(self, change_class_weights={'use': False, 'epoch': -1, 'class_weights': None}):
        self.change_class_weights = change_class_weights

    def _before_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch: DATA_BATCH = None,
                         mode: str = 'val') -> None:     
        
        current_iter = runner.iter
        iters_per_epoch = len(runner.train_dataloader)
        current_epoch = current_iter // iters_per_epoch
        
        if self.change_class_weights['use'] and current_epoch == self.change_class_weights['epoch'] and self.change_class_weights['epoch'] is not None:
            
            if len(self.change_class_weights) == 1:
                class_weights = self.change_class_weights
            elif isinstance(self.change_class_weights, float):
                class_weights = [self.change_class_weights]
            elif len(self.change_class_weights) == len(runner.train_dataloader.dataset.CLASSES):
                class_weights = self.change_class_weights
            else:
                NotImplementedError(f'NOT Considered this case of class-weight to change class-weights: {self.change_class_weights}')
                        
            change_class_weights_to_loss_decode(runner, class_weights)
        
        
                
            
            
            
        