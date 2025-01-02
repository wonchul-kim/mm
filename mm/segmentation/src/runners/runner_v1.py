from mmengine.runner import Runner
import copy 
from typing import Callable, Dict, List, Optional, Sequence, Union
from torch.utils.data import DataLoader
from threading import Thread


class RunnerV1(Runner):
    
    @classmethod
    def build_dataloader(cls, dataloader: Union[DataLoader, Dict],
                         seed: Optional[int] = None,
                         diff_rank_seed: bool = False) -> DataLoader:
        
        vis_dataloader_ratio = dataloader.vis_dataloader_ratio
        vis_dir = dataloader.vis_dir 
        del dataloader.vis_dataloader_ratio
        del dataloader.vis_dir
        
        data_loader = super().build_dataloader(dataloader, seed, diff_rank_seed)
        if vis_dataloader_ratio:
            from mm.segmentation.utils.visualizers import vis_dataloader
            
            if 'train' in dataloader.dataset.data_prefix['img_path']:
                mode = 'train'
            elif 'val' in dataloader.dataset.data_prefix['img_path']:
                mode = 'val'
            elif 'test' in dataloader.dataset.data_prefix['img_path']:
                mode = 'test'
            else:
                raise NotImplementedError(f'CANNOT tell model b/t train, val and test')
            
            # FIXME: any other way not to build ???
            # vis_dataloader(super().build_dataloader(dataloader, seed, diff_rank_seed), mode,
            #                ratio=vis_dataloader_ratio, output_dir=vis_dir)
            Thread(target=vis_dataloader, args=(super().build_dataloader(dataloader, seed, diff_rank_seed), mode,
                                                vis_dataloader_ratio, vis_dir), daemon=True).start()
        
        return data_loader