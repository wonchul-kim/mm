
from .cosnet import COSNet

__all__ = ['COSNet']


backbone_weights_map = {
    'upernet-r50': 'ade20k_iter_160000',
}