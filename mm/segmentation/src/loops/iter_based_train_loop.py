import bisect
import logging
import time
from typing import Dict, List, Optional, Sequence, Tuple, Union
from mmengine.optim import OptimWrapper

import torch
from torch.utils.data import DataLoader

from mmengine.evaluator import Evaluator
from mmengine.logging import HistoryBuffer, print_log
from mmengine.registry import LOOPS

from mmengine.runner.loops import IterBasedTrainLoop,  _update_losses

@LOOPS.register_module()
class IterBasedTrainLoopV2(IterBasedTrainLoop):
    def __init__(
        self,
        runner,
        dataloader: Union[DataLoader, Dict],
        max_iters: int,
        evaluator: Union[Evaluator, Dict, List] = None,
        val_begin: int = 1,
        val_interval: int = 1000,
        dynamic_intervals: Optional[List[Tuple[int, int]]] = None) -> None:
        
        super().__init__(runner, 
                         dataloader, 
                         max_iters, 
                         val_begin, 
                         val_interval,
                         dynamic_intervals)
        
        if evaluator is not None:
            if isinstance(evaluator, (dict, list)):
                self.evaluator = runner.build_evaluator(evaluator)  # type: ignore
            else:
                assert isinstance(evaluator, Evaluator), (
                    'evaluator must be one of dict, list or Evaluator instance, '
                    f'but got {type(evaluator)}.')
                self.evaluator = evaluator  # type: ignore
            if hasattr(self.dataloader.dataset, 'metainfo'):
                self.evaluator.dataset_meta = self.dataloader.dataset.metainfo
                self.runner.visualizer.dataset_meta = \
                    self.dataloader.dataset.metainfo
            else:
                print_log(
                    f'Dataset {self.dataloader.dataset.__class__.__name__} has no '
                    'metainfo. ``dataset_meta`` in evaluator, metric and '
                    'visualizer will be None.',
                    logger='current',
                    level=logging.WARNING)
        else:
            self.evaluator = None
            
        self.train_loss: Dict[str, HistoryBuffer] = dict()
        self.train_loss.clear()
        
    @torch.no_grad()
    def run_eval_iter(self, data_batch: Sequence[dict]):
        self.runner.model.eval()
        outputs = self.runner.model.val_step(data_batch)
        outputs, self.train_loss = _update_losses(outputs, self.train_loss)
        self.evaluator.process(data_samples=outputs, data_batch=data_batch)