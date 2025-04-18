from typing import Optional, Sequence

from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmyolo.registry import HOOKS

from mm.yolo.utils.visualizers.vis_test import vis_test

@HOOKS.register_module()
class VisualizeTest(Hook):
    def __init__(self, output_dir, classes=None, annotate=False, conf_threshold=0.2,  
                 save_raw=False):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.output_dir = output_dir
        self.annotate = annotate 
        self.conf_threshold = conf_threshold
        self.save_raw = save_raw
        self.classes = classes

    def after_test_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Optional[Sequence],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        vis_test(outputs=outputs, output_dir=self.output_dir, data_batch=data_batch, 
                 batch_idx=batch_idx,  
                 annotate=self.annotate,  
                 conf_threshold=self.conf_threshold,
                 save_raw=self.save_raw,
                 classes=self.classes,
                 )



