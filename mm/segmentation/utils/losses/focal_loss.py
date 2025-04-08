import torch 
import torch.nn.functional as F
import torch.nn as nn 
from mmseg.registry import MODELS

@MODELS.register_module()
class SoftmaxFocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25, loss_weight=1.0,
                 loss_name_: str = 'loss_focal'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.loss_weight = loss_weight
        self.loss_name_ = loss_name_

    def forward(self, pred, target, **kwargs):
        # pred: [N, C, H, W], target: [N, H, W]
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return self.loss_weight * focal_loss.mean()
    
    @property
    def loss_name(self):
        return self.loss_name_