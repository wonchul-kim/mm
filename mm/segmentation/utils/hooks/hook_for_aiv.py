import numpy as np
from typing import Optional, Sequence, Union

from mmseg.registry import HOOKS
from mmengine.hooks import Hook

DATA_BATCH = Optional[Union[dict, tuple, list]]

@HOOKS.register_module()
class HookForAiv(Hook):
    def __init__(self, aiv, **kwargs):
        self.aiv = aiv
        
    def before_run(self, runner, metrics) -> None:

        ## set custom logger, monitor
        if self.aiv:
            pass