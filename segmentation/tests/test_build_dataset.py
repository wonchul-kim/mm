
from mmengine.config import Config
from mmseg.registry import DATASETS
import torch

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parent

def main():
    
    cfg = Config.fromfile(ROOT / 'segmentatoin/configs/_base_/datasets/mask.py')

    dataset = DATASETS.build(cfg)

if __name__ == '__main__':
    main()