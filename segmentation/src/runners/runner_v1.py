from mmengine.runner import Runner
import copy 
from typing import Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import DataLoader

class RunnerV1(Runner):
    
    @classmethod
    def build_dataloader(cls, dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        
        data_loader = super().build_dataloader(dataloader, seed, diff_rank_seed)
        
        
        vis_dataloader = True
        if vis_dataloader:
            from segmentation.utils.visualizers import vis_dataloader
            
            if 'train' in dataloader.dataset.data_prefix['img_path']:
                mode = 'train'
            elif 'val' in dataloader.dataset.data_prefix['img_path']:
                mode = 'val'
            elif 'test' in dataloader.dataset.data_prefix['img_path']:
                mode = 'test'
            else:
                raise NotImplementedError(f'CANNOT tell model b/t train, val and test')
            
            # FIXME: any other way not to build ???
            vis_dataloader(super().build_dataloader(dataloader, seed, diff_rank_seed), mode)
        
        return data_loader