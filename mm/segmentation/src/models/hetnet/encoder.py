from .resnext.resnext101_regular import ResNeXt101

from mmseg.registry import MODELS
from mmengine.model import BaseModule


@MODELS.register_module()
class HetNetEncoder(BaseModule):
    def __init__(self, frozen_stages=2, weights=None):
        super().__init__()
        self.frozen_stages = frozen_stages
        self.bkbone = ResNeXt101(weights)
        
        self._freeze_stages()

    def forward(self, x): 

        return self.bkbone(x)
        
    def _freeze_stages(self):
        
        if self.frozen_stages != -1:
            for number in range(0, self.frozen_stages + 1):
                layer = getattr(self.bkbone, f'layer{number}')
                layer.eval()
                print(f"Freeze layer: {number} =====================================")
                for name, param in layer.named_parameters():
                    param.requires_grad = False
                    print(f"Freeze {name} to param.requires_grad({param.requires_grad})")
                        
                