from .resnext.resnext101_regular import ResNeXt101

from mmseg.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class HetNetEncoder(BaseModule):
    def __init__(self, weights='/HDD/weights/hetnet/resnext_101_32x4d.pth'):
        super().__init__()
        self.bkbone = ResNeXt101(weights)
        
    def forward(self, x): 

        return self.bkbone(x)
        