from mmseg.registry import MODELS
from mmseg.utils import OptConfigType
from mmseg.models.backbones.pidnet import PIDNet as BasePIDNet

@MODELS.register_module(force=True)
class PIDNet(BasePIDNet):
    def __init__(self,
                 in_channels: int = 3,
                 channels: int = 64,
                 ppm_channels: int = 96,
                 num_stem_blocks: int = 2,
                 num_branch_blocks: int = 3,
                 align_corners: bool = False,
                 norm_cfg: OptConfigType = dict(type='BN'),
                 act_cfg: OptConfigType = dict(type='ReLU', inplace=True),
                 init_cfg: OptConfigType = None,
                 frozen_stages: int = -1,
                 **kwargs):
        super().__init__(in_channels=in_channels,
                 channels=channels,
                 ppm_channels=ppm_channels,
                 num_stem_blocks=num_stem_blocks,
                 num_branch_blocks=num_branch_blocks,
                 align_corners=align_corners,
                 norm_cfg=norm_cfg,
                 act_cfg=act_cfg,
                 init_cfg=init_cfg,
                 frozen_stages=frozen_stages,
                 **kwargs)

        self.frozen_stages = frozen_stages
        self._freeze_stages()
    


    def _freeze_stages(self) -> None:
        # for frozen_idx in range(self.frozen_stages + 1):
        #     # freeze downsample_layers
        #     for param in self.downsample_layers[frozen_idx].parameters():
        #         param.requires_grad = False

        #     # freeze stages
        #     self.stages[frozen_idx].eval()
        #     for param in self.stages[frozen_idx].parameters():
        #         param.requires_grad = False
        if self.frozen_stages != -1:
            self.eval()
            for param in self.parameters():
                param.requires_grad = False
                
                
if __name__ == '__main__':
    model = PIDNet(frozen_stages=1)
    
    model._freeze_stages()


    for param in model.parameters():
        print(param.requires_grad)
    
    