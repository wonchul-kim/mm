import torch
import random
import unittest
import numpy as np
from mm.segmentation.src.models.lps.layers import(PolyphaseInvariantDown2D,LPS,get_logits_model,
                   get_antialias)
from mm.segmentation.src.models.lps.layers.polyup import(PolyphaseInvariantUp2D,LPS_u)
from mm.segmentation.src.models.lps.base_segmentation import DDAC_MODEL_MAP, DDACSegmentation
from functools import partial
from mm.segmentation.src.models.lps.models import get_model as get_backbone

from mmseg.registry import MODELS 
from mmengine.model import BaseModule

@MODELS.register_module()
class LPSEncoder(BaseModule):
    def __init__(self, num_classes: int):
        super().__init__()
            
        # Antialias pars
        antialias_mode = 'DDAC'
        antialias_size = 3
        antialias_padding = 'same'
        antialias_padding_mode = 'circular'
        antialias_group = 1
        unpool_antialias_scale = 2

        # Antialias filters
        antialias = get_antialias(antialias_mode=antialias_mode,
                                antialias_size=antialias_size,
                                antialias_padding=antialias_padding,
                                antialias_padding_mode=antialias_padding_mode,
                                antialias_group=antialias_group)
        unpool_antialias = get_antialias(antialias_mode=antialias_mode,
                                        antialias_size=antialias_size,
                                        antialias_padding=antialias_padding,
                                        antialias_padding_mode=antialias_padding_mode,
                                        antialias_group=antialias_group,
                                        antialias_scale=unpool_antialias_scale)

        # Pooling layer
        get_logits = get_logits_model('LPSLogitLayers')
        pool_layer = partial(PolyphaseInvariantDown2D,
                            component_selection=LPS,
                            get_logits=get_logits,
                            antialias_layer=antialias)

        # Unpool layer
        unpool_layer = partial(PolyphaseInvariantUp2D,
                                component_selection=LPS_u,
                                antialias_layer=unpool_antialias)

        # Extra backbone args
        extras_model = {
            'logits_channels': None,
            'conv1_stride': False,
            'maxpool_zpad': False,
            'swap_conv_pool': False,
            'inc_conv1_support': True,
            'apply_maxpool': True,
            'ret_prob': True,
            'forward_pool_method': 'LPS',
            'forward_ret_prob_logits': False,
        }

        # Backbone
        backbone = get_backbone('ResNet18Custom')(
            input_shape=(),
            num_classes=num_classes,
            padding_mode='circular',
            pooling_layer=pool_layer,
            extras_model=extras_model
        )

        # # state_dict = torch.load("/HDD/weights/lps/ResNet18_LPS_circular_basic.ckpt")
        # # state_dict = torch.load("/HDD/weights/lps/ResNet18_LPS_DDAC3_circular_basic.ckpt")
        # state_dict = torch.load("/HDD/weights/lps/ResNet18_LPS_LPF2_circular_basic.ckpt")
        # backbone.load_state_dict(state_dict)

        # Segmenter
        self.model = DDACSegmentation(
            model_name='deeplabv3plus_resnet_lps_unpool',
            num_classes=num_classes,
            # output_stride=8, 
            output_stride=16,
            backbone=backbone,
            unpool_layer=unpool_layer,
            classifier_padding_mode='circular')


        
    def forward(self, x):
        
        return self.model(x)

