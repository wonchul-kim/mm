from mmengine.logging import MMLogger, print_log

from mmseg.registry import MODELS
from mmseg.models.backbones.mit import MixVisionTransformer as BaseMixVisionTransformer

@MODELS.register_module(force=True)
class MixVisionTransformer(BaseMixVisionTransformer):
    def __init__(self,
                 in_channels=3,
                 embed_dims=64,
                 num_stages=4,
                 num_layers=[3, 4, 6, 3],
                 num_heads=[1, 2, 4, 8],
                 patch_sizes=[7, 3, 3, 3],
                 strides=[4, 2, 2, 2],
                 sr_ratios=[8, 4, 2, 1],
                 out_indices=(0, 1, 2, 3),
                 mlp_ratio=4,
                 qkv_bias=True,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.,
                 act_cfg=dict(type='GELU'),
                 norm_cfg=dict(type='LN', eps=1e-6),
                 pretrained=None,
                 init_cfg=None,
                 with_cp=False,
                 frozen_stages=-1):
        
        super().__init__(in_channels=in_channels,
                 embed_dims=embed_dims,
                 num_stages=num_stages,
                 num_layers=num_layers,
                 num_heads=num_heads,
                 patch_sizes=patch_sizes,
                 strides=strides,
                 sr_ratios=sr_ratios,
                 out_indices=out_indices,
                 mlp_ratio=mlp_ratio,
                 qkv_bias=qkv_bias,
                 drop_rate=drop_rate,
                 attn_drop_rate=attn_drop_rate,
                 drop_path_rate=drop_path_rate,
                 act_cfg=act_cfg,
                 norm_cfg=norm_cfg,
                 pretrained=pretrained,
                 init_cfg=init_cfg,
                 with_cp=with_cp,
                 )

        self.frozen_stages = frozen_stages

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super().train(mode)
        self._freeze_stages()


    def _freeze_stages(self):
        logger: MMLogger = MMLogger.get_current_instance()
        print_log(f"* Frozen-stages are {self.frozen_stages}", logger)
        
        if self.frozen_stages >= 0:
            
            for idx in range(self.frozen_stages):
                print_log(f"* Frozing the {idx} layer", logger)
                layer = self.layers[idx]
                layer.eval()
                for jdx, param in enumerate(layer.parameters()):
                    param.requires_grad = False
                    
                print_log(f"   : {jdx} params are frozen", logger)
                
            