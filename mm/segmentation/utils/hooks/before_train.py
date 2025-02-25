import numpy as np
from typing import Optional, Sequence, Union
from copy import deepcopy
from threading import Thread

from mmseg.registry import HOOKS
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookBeforeTrain(Hook):
    def __init__(self, **kwargs):
        self.vis_dataloader_dir = kwargs['debug_dir']
        self.vis_dataloader_ratio = kwargs['ratio']
        
    def before_train(self, runner) -> None:
        if self.vis_dataloader_ratio:
            from mm.segmentation.utils.visualizers import vis_dataloader
            
            # vis_dataloader(runner.train_dataloader.dataset, 'train',
            #                                     self.vis_dataloader_ratio, self.vis_dataloader_dir)
            # vis_dataloader(runner.val_dataloader, 'val',
            #                                     self.vis_dataloader_ratio, self.vis_dataloader_dir)
            Thread(target=vis_dataloader, args=(deepcopy(runner.train_dataloader.dataset), 'train',
                                                self.vis_dataloader_ratio, self.vis_dataloader_dir), daemon=True).start()
            if hasattr(runner, 'val_dataloader'):
                Thread(target=vis_dataloader, args=(deepcopy(runner.val_dataloader), 'val',
                                                self.vis_dataloader_ratio, self.vis_dataloader_dir), daemon=True).start()