import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List 

from sam2.build_sam import build_sam2
from mmseg.registry import MODELS
from mmengine.model import BaseModule
from mmseg.utils import OptConfigType
from .modules import DoubleConv, Up, Adapter, BasicConv2d, RFB_modified

# class SAM2UNet(BaseModule):
@MODELS.register_module()
class SAM2UNet(nn.Module):
    def __init__(self, num_classes, 
                    model_cfg: str,
                    exclude_layers: List[str] = ['sam_mask_decoder', 'sam_prompt_encoder', 'memory_encoder',
                                                 'memory_attention', 'mask_downsample', 'obj_ptr_tpos_proj', 
                                                 'obj_ptr_proj', 'image_encoder.neck'],
                    forzen_stages: int=1,
                    checkpoint_path=None) -> None:
        super().__init__()    

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

        if forzen_stages != 0:
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
        
        
        self.rfb1 = RFB_modified(144, 64)
        self.rfb2 = RFB_modified(288, 64)
        self.rfb3 = RFB_modified(576, 64)
        self.rfb4 = RFB_modified(1152, 64)
        self.up1 = (Up(128, 64))
        self.up2 = (Up(128, 64))
        self.up3 = (Up(128, 64))
        self.up4 = (Up(128, 64))
        self.side1 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.side2 = nn.Conv2d(64, num_classes, kernel_size=1)
        self.head = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.side1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.side2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        
        return out, out1, out2 if self.training else out

    def forward_export(self, x):
        x1, x2, x3, x4 = self.encoder(x)
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        out = F.interpolate(self.head(x), scale_factor=4, mode='bilinear')
        
        return out

if __name__ == "__main__":
    with torch.no_grad():
        model = SAM2UNet(4).cuda()
        x = torch.randn(1, 3, 352, 352).cuda()
        out, out1, out2 = model(x)
        print(out.shape, out1.shape, out2.shape)