from typing import Tuple, Union
from torch import Tensor
import torch.nn as nn

from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList

def structure_loss(pred, mask, num_classes):
    import torch
    import torch.nn.functional as F
    mask = mask.squeeze(1)  # 채널 차원 제거
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask.float(), kernel_size=31, stride=1, padding=15) - mask.float())
    ce_loss = F.cross_entropy(pred, mask.long(), reduction='none')
    ce_loss = (weit * ce_loss).sum(dim=(1, 2)) / weit.sum(dim=(1, 2))
    
    return ce_loss.mean()

@MODELS.register_module()
class SAM2UNetHead(BaseDecodeHead):
    def __init__(self,
                 in_channels: int,
                 channels: int,
                 num_classes: int,
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
        if self.training:
            return inputs
        else:
            return inputs[0]

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                    batch_data_samples: SampleList) -> dict:
        loss = dict()
        seg_label = self._stack_batch_gt(batch_data_samples)
        
        pred0, pred1, pred2 = seg_logits 
        
        loss['loss0'] = structure_loss(pred0, seg_label, self.num_classes)
        loss['loss1'] = structure_loss(pred1, seg_label, self.num_classes)
        loss['loss2'] = structure_loss(pred2, seg_label, self.num_classes)
        loss['loss'] = loss['loss0'] + loss['loss1'] + loss['loss2']
        
        return loss
        
                    
                    
            