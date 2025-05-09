# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import warnings
from typing import Optional, Sequence

import mmcv
import mmengine.fileio as fileio
from mmengine.hooks import Hook
from mmengine.runner import Runner
from mmengine.visualization import Visualizer

from mmseg.registry import HOOKS
from mmseg.structures import SegDataSample

from mm.segmentation.utils.visualizers.vis_test import vis_test

@HOOKS.register_module()
class VisualizeTest(Hook):
    def __init__(self, output_dir, annotate=False, contour_thres=10, contour_conf=0.5, 
                 save_raw=False, legend=True):
        self._visualizer: Visualizer = Visualizer.get_current_instance()
        self.output_dir = output_dir
        self.annotate = annotate 
        self.contour_thres = contour_thres
        self.contour_conf = contour_conf
        self.save_raw = save_raw
        self.legend = legend

    def _after_iter(self,
                    runner: Runner,
                    batch_idx: int,
                    data_batch: dict,
                    outputs: Sequence[SegDataSample],
                    mode: str = 'val') -> None:
        """Run after every ``self.interval`` validation iterations.

        Args:
            runner (:obj:`Runner`): The runner of the validation process.
            batch_idx (int): The index of the current batch in the val loop.
            data_batch (dict): Data from dataloader.
            outputs (Sequence[:obj:`SegDataSample`]): Outputs from model.
            mode (str): mode (str): Current mode of runner. Defaults to 'val'.
        """
        vis_test(outputs, self.output_dir, data_batch, batch_idx, 
                 annotate=self.annotate, contour_thres=self.contour_thres, 
                 contour_conf=self.contour_conf, save_raw=self.save_raw,
                 legend=self.legend)



