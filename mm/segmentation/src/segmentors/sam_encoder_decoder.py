import torch 
from typing import Dict, Union

from mmseg.registry import MODELS
from mmseg.models import EncoderDecoder
from mmengine.optim import OptimWrapper
from mm.segmentation.src.optimizers.sam_optimizer import disable_running_stats, enable_running_stats


@MODELS.register_module()
class SAMEncoderDecoder(EncoderDecoder):

    def train_step(self, data: Union[dict, tuple, list],
                   optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
        # Enable automatic mixed precision training context.
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        disable_running_stats(self)
        with optim_wrapper.optim_context(self):
            data = self.data_preprocessor(data, True)
            losses = self._run_forward(data, mode='loss')  # type: ignore
        parsed_losses, log_vars = self.parse_losses(losses)  # type: ignore
        optim_wrapper.update_params(parsed_losses)
        enable_running_stats(self)
        return log_vars