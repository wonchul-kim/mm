import torch
from torch import Tensor
import torch.distributed as dist

from mmseg.utils import SampleList
from mmseg.models.decode_heads.segformer_head import SegformerHead as BaseSegformerHead
from mmseg.registry import MODELS

from mm.segmentation.src.datasets.infobatch_dataset import *


@MODELS.register_module(force=True)
class SegformerHead(BaseSegformerHead):

    def loss_by_feat(self, seg_logits: Tensor,
                     batch_data_samples: SampleList) -> dict:

        loss = super().loss_by_feat(seg_logits, batch_data_samples)

        ### infobatch ################################################################################################
        if loss['loss_ce'].ndim == 3 and hasattr(self, 'dataset') and self.dataset is not None:
            loss_ce_persample = loss['loss_ce'].mean(dim=(1, 2))

            indices, weights = [], []
            for batch_data_sample in batch_data_samples:
                assert 'index' in batch_data_sample and 'weight' in batch_data_sample, RecursionError(f'There is no weight and index in data_sample')
                indices.append(batch_data_sample.index)
                weights.append(batch_data_sample.weight)

            indices = torch.tensor(indices, dtype=torch.int)
            weights = torch.tensor(weights)

            with torch.no_grad():
                scores = loss_ce_persample
                if dist.is_available() and dist.is_initialized():
                    low,high = split_index(indices)
                    low,high = low.cuda(),high.cuda()
                    tuple = torch.stack([low,high,scores])
                    tuple_all = concat_all_gather(tuple, dim=1)
                    low_all, high_all, scores_all = tuple_all[0].type(torch.int), tuple_all[1].type(torch.int), tuple_all[2]
                    indices_all = recombine_index(low_all,high_all)
                    # self.dataset.__setscore__(indices_all.detach().cpu().numpy(), scores_all.detach().cpu().numpy())
                    self.dataset.__setscore__(indices_all.numpy(), scores_all.detach().cpu().numpy())
                else:
                    self.dataset.__setscore__(indices.numpy(), scores.detach().cpu().numpy())

            loss['loss_ce'] = (loss_ce_persample * weights.to(loss_ce_persample.device)).mean()
        #############################################################################################################

        return loss