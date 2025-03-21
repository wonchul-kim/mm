import numpy as np
from typing import Optional, Sequence, Union

from mmseg.registry import HOOKS
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookAfterValEpoch(Hook):
    def __init__(self, **kwargs):
        pass        

    def after_val_epoch(self, runner, metrics) -> None:

        pass