from typing import Tuple, Union, List
from torch import Tensor
import torch.nn as nn

from mmseg.models.utils import resize
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from mmseg.registry import MODELS
from mmseg.utils import OptConfigType, SampleList, ConfigType

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)
    
    
class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class Adapter(nn.Module):
    def __init__(self, blk) -> None:
        super(Adapter, self).__init__()
        self.block = blk
        dim = blk.attn.qkv.in_features
        self.prompt_learn = nn.Sequential(
            nn.Linear(dim, 32),
            nn.GELU(),
            nn.Linear(32, dim),
            nn.GELU()
        )

    def forward(self, x):
        prompt = self.prompt_learn(x)
        promped = x + prompt
        net = self.block(promped)
        return net
    

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x
    

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x


# def structure_loss(pred, mask):
#     import torch
#     import torch.nn.functional as F
#     mask = mask.squeeze(1) 
#     weit = 1 + 5*torch.abs(F.avg_pool2d(mask.float(), kernel_size=31, stride=1, padding=15) - mask.float())
#     ce_loss = F.cross_entropy(pred, mask.long(), reduction='none')
#     ce_loss = (weit * ce_loss).sum(dim=(1, 2)) / weit.sum(dim=(1, 2))
    
#     return ce_loss.mean()

def structure_loss(pred, mask, ce):
    import torch
    import torch.nn.functional as F
    mask = mask.squeeze(1) 
    weit = 1 + 5*torch.abs(F.avg_pool2d(mask.float(), kernel_size=31, stride=1, padding=15) - mask.float())
    ce_loss = ce(pred, mask.long())
    ce_loss = (weit * ce_loss).sum(dim=(1, 2)) / weit.sum(dim=(1, 2))
    
    return ce_loss.mean()

@MODELS.register_module()
class SAM2UNetHead(BaseDecodeHead):
    def __init__(self,
                 num_classes: int,
                 in_channels: int = 3,
                 channels: int = 64,
                 rfb_channels: List[int] = [[144, 64], [288, 64], [576, 64], [1152, 64]], # large
                 up_channels: List[int] = [[128, 64], [128, 64], [128, 64]],
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
        
        # RFB
        self.rfb1 = RFB_modified(rfb_channels[0][0], rfb_channels[0][1])
        self.rfb2 = RFB_modified(rfb_channels[1][0], rfb_channels[1][1])
        self.rfb3 = RFB_modified(rfb_channels[2][0], rfb_channels[2][1])
        self.rfb4 = RFB_modified(rfb_channels[3][0], rfb_channels[3][1])
        
        # Upsample
        self.up1 = (Up(up_channels[0][0], up_channels[0][1]))
        self.up2 = (Up(up_channels[1][0], up_channels[1][1]))
        self.up3 = (Up(up_channels[2][0], up_channels[2][1]))
        
        # Conv
        self.conv1 = nn.Conv2d(up_channels[0][1], num_classes, kernel_size=1)
        self.conv2 = nn.Conv2d(up_channels[1][1], num_classes, kernel_size=1)
        self.conv3 = nn.Conv2d(up_channels[2][1], num_classes, kernel_size=1)

    def forward(self, inputs: Union[Tensor, Tuple[Tensor]]) -> Union[Tensor, Tuple[Tensor]]:
        x1, x2, x3, x4 = inputs
        x1, x2, x3, x4 = self.rfb1(x1), self.rfb2(x2), self.rfb3(x3), self.rfb4(x4)
        x = self.up1(x4, x3)
        out1 = F.interpolate(self.conv1(x), scale_factor=16, mode='bilinear')
        x = self.up2(x, x2)
        out2 = F.interpolate(self.conv2(x), scale_factor=8, mode='bilinear')
        x = self.up3(x, x1)
        out = F.interpolate(self.conv3(x), scale_factor=4, mode='bilinear')
        
        return (out, out1, out2) if self.training else out

    def loss_by_feat(self, seg_logits: Tuple[Tensor],
                    batch_data_samples: SampleList) -> dict:
        loss = dict()
        seg_label = self._stack_batch_gt(batch_data_samples)
        
        pred0, pred1, pred2 = seg_logits 
        
        loss['loss0'] = structure_loss(pred0, seg_label, self.loss_decode[0])
        loss['loss1'] = structure_loss(pred1, seg_label, self.loss_decode[0])
        loss['loss2'] = structure_loss(pred2, seg_label, self.loss_decode[0])
        loss['loss'] = loss['loss0'] + loss['loss1'] + loss['loss2']
        
        return loss
        
                    
                    