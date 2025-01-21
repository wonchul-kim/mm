

from typing import Optional, Sequence, Union

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
        
        
        ## evaluate
        if runner.train_loop.evaluator:
            runner.train_loop.run_eval_iter(data_batch)

            if batch_idx != 0 and batch_idx%len(runner.train_dataloader) == 0:
                metrics = runner.train_loop.evaluator.evaluate(len(runner.train_dataloader.dataset))

                if runner.train_loop.train_loss:
                    loss_dict = _parse_losses(runner.train_loop.train_loss, 'train')
                    metrics.update(loss_dict)
                    runner.train_loop.train_loss.clear()
        
        ## log
        if batch_idx != 0 and batch_idx%len(runner.train_dataloader) == 0:
            train_log = {}
            if 'loss' in outputs:
                train_log['loss'] = outputs['loss'].item()
                
            for idx, base_lr in enumerate(runner.optim_wrapper.get_lr()['base_lr']):
                train_log[f'lr_{idx}'] = base_lr
            
            runner.train_log = train_log
            
        # TODO: metrics for train dataset
            
            
            
            
        