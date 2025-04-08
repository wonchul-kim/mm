from typing import Tuple, Union
from torch import Tensor

from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType


@MODELS.register_module()
class EmptyDecodeHead(BaseDecodeHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 channels: int = 64,
                 norm_cfg: OptConfigType = dict(type='BN', requires_grad=True),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 **kwargs):
        super().__init__(
            in_channels=in_channels,
            channels=channels,
            num_classes=num_classes,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg,
            **kwargs)
        
    def forward(self, inputs: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        
        return inputs


                    