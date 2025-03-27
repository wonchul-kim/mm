from typing import Optional, Sequence, Union

from mmseg.registry import HOOKS
from mmengine.hooks import Hook
from mmengine.logging import print_log
from mm.segmentation.utils.visualizers.vis_val import vis_val

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class VisualizeVal(Hook):
    def __init__(self, freq_epoch, ratio, output_dir, **kwargs):
        self.freq_epoch = freq_epoch
        self.ratio = ratio 
        self.output_dir = output_dir
        

    def after_val_iter(self,
                       runner,
                       batch_idx: int,
                       data_batch: DATA_BATCH = None,
                       outputs: Optional[Sequence] = None) -> None:
        
        
        current_iter = runner.iter
        iters_per_epoch = len(runner.train_dataloader)
        current_epoch = current_iter // iters_per_epoch
            
        if current_epoch%self.freq_epoch == 0:
            vis_val(outputs, self.ratio, self.output_dir, current_epoch, batch_idx)
