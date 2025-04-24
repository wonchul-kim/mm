

from typing import Optional, Union
import numpy as np
from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from mm.segmentation.utils.class_weights import (get_class_frequency_v2, get_class_weights, 
                                                 get_total_class_frequency, apply_class_weights_to_loss_decode)

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookBeforeRun(Hook):
    def __init__(self, apply_class_weights=False, 
                        class_frequency=None, class_weights=None, ignore_background=True, 
                        *args, **kwargs):
        self.apply_class_weights = apply_class_weights
        self.class_frequency = class_frequency
        self.class_weights = class_weights
        self.ignore_background = ignore_background

    def before_run(self, runner) -> None:
        
        if not self.apply_class_weights:
            self.class_frequency = None 
            self.class_weights = None
        else:
            if self.class_frequency is None or self.class_frequency == []:
                self.class_frequency = np.zeros(len(runner.train_dataloader.dataset.CLASSES), dtype=np.int64)

            for batch in runner.train_dataloader.dataset:
                mask = batch['data_samples'].gt_sem_seg.data.numpy()[0]
                class_freq = get_class_frequency_v2(mask, len(runner.train_dataloader.dataset.CLASSES))
                self.class_frequency += class_freq
                
            self.class_weights = get_class_weights(self.class_frequency, ignore_background=self.ignore_background)
            apply_class_weights_to_loss_decode(runner, self.class_weights)

            
            

            
                
                
        
                
            
            
            
        