import torch.nn as nn
from typing import List 

from mmseg.registry import MODELS
from mmengine.model import BaseModule
from .decode_heads.sam2unet_head import Adapter

@MODELS.register_module()
class SAM2Encoder(BaseModule):
# class SAM2Encoder(nn.Module):
    def __init__(self, model_cfg: str,
                    exclude_layers: List[str] = ['sam_mask_decoder', 'sam_prompt_encoder', 'memory_encoder',
                                                 'memory_attention', 'mask_downsample', 'obj_ptr_tpos_proj', 
                                                 'obj_ptr_proj', 'image_encoder.neck'],
                    frozen_stages: int=1,
                    checkpoint_path=None) -> None:
        super().__init__()   
        try: 
            from sam2.build_sam import build_sam2
        except:
            from mm.utils.git import install_by_clone
            print(f"CANNOT import sam2, need to install it first")
            install_by_clone(url="https://github.com/facebookresearch/sam2.git", dir_name='sam2')
            print(f"SUCCESSFULLY Installed sam2")
            from sam2.build_sam import build_sam2

        if checkpoint_path:
            model = build_sam2(model_cfg, checkpoint_path)
            print(f"Build SMA2 with checkpoint: {checkpoint_path}")
        else:
            model = build_sam2(model_cfg)
            print(f"Build SMA2 without checkpoint")
            
        for layer_name in exclude_layers:
            if '.' in layer_name:  # Handle nested attributes
                parent, child = layer_name.split('.')
                if hasattr(getattr(model, parent), child):
                    delattr(getattr(model, parent), child)
            elif hasattr(model, layer_name):
                delattr(model, layer_name)

        self.encoder = model.image_encoder.trunk

        if frozen_stages != -1:
            for param in self.encoder.parameters():
                param.requires_grad = False
            blocks = []
            for block in self.encoder.blocks:
                blocks.append(
                    Adapter(block)
                )
                
            print(f"FREEZE the SAM2 encoder")
                
        self.encoder.blocks = nn.Sequential(
            *blocks
        )
        
    def forward(self, x):
        return self.encoder(x)
