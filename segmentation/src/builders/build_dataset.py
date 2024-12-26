
from mmengine.config import Config
from mmseg.registry import DATASETS
import segmentation.src


if __name__ == '__main__':
    from pathlib import Path 
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]

    cfg = Config.fromfile(ROOT / 'configs/_base_/datasets/labelme.py')
    
    
    data_root = "/HDD/datasets/projects/LX/24.12.12/split_mask_patch_dataset"
    img_suffix='.png'
    seg_map_suffix='.png'
    # classes=('background', 'timber', 'screw')
    classes=('timber', 'screw')
    batch_size = 1
    # set dataset ====================================================================================================
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.dataset['data_root'] = data_root
    cfg.train_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
    cfg.train_dataloader.dataset['classes'] = classes
    cfg.train_dataloader.dataset['img_suffix'] = img_suffix
    
    cfg.val_dataloader.batch_size = batch_size
    cfg.val_dataloader.dataset['data_root'] = data_root
    cfg.val_dataloader.dataset['classes'] = classes
    cfg.val_dataloader.dataset['img_suffix'] = img_suffix
    cfg.val_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
    

    dataset = DATASETS.build(cfg.train_dataloader.dataset)