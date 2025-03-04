import torch
import torch.nn.functional as F

from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
from mmengine.structures import PixelData



def translate_tensor(tensor, shift_x, shift_y):
    _, h, w = tensor.shape
    pad_x = abs(shift_x)
    pad_y = abs(shift_y)
    
    # 패딩 적용
    padded = F.pad(tensor, (pad_x, pad_x, pad_y, pad_y), mode='constant', value=0)
    
    # 이동된 영역 추출
    if shift_y >= 0:
        start_y = pad_y*2
        end_y = start_y + h
    else:
        start_y = 0
        end_y = start_y + h
    if shift_x >= 0:
        start_x = pad_x*2
        end_x = start_x + w
    else:
        start_x = 0
        end_x = start_x + w
    
    translated = padded[:, start_y:end_y, start_x:end_x]
    
    return translated

class TTASegModel(torch.nn.Module):
    def __init__(self, model, augs):
        super().__init__()
        self._model = model 
        self._augs = augs
    
    def test_step(self, data_batch):
        return self._model.test_step(data_batch)
            
    def run_tta_batch(self, data_batch):
        batch_inputs = data_batch['inputs']
        batch_data_samples = data_batch['data_samples']
        
        new_data_batch = {'inputs': [], 'data_samples': []}
        batch_outputs = []
        for batch_input, batch_data_sample in zip(batch_inputs, batch_data_samples):
            self.make_tta_batch(new_data_batch, batch_input, batch_data_sample)
            batch_output = self._model.test_step(new_data_batch)
            self.recover_tta_output(batch_output)
            batch_outputs.append(self.merge_tta_batch(batch_output))
        return batch_outputs
            
    @property
    def model(self):
        return self._model 

    @property
    def augs(self):
        return self._augs 
    
    def __getattr__(self, name):
        if name in ['test_step', '_model']:
            return super().__getattr__(name)
        else:
            return getattr(self._model, name)

    def make_tta_batch(self, new_data_batch, batch_input, batch_data_sample):
        new_data_batch['inputs'].append(batch_input)
        new_data_batch['data_samples'].append(batch_data_sample)
        for key, val in self._augs.items():
            if key == 'HorizontalFlip' and val:
                new_data_batch['inputs'].append(RandomHorizontalFlip(1)(batch_input))
                new_data_batch['data_samples'].append(batch_data_sample)
            
            if key == 'VerticalFlip' and val:
                new_data_batch['inputs'].append(RandomVerticalFlip(1)(batch_input))
                new_data_batch['data_samples'].append(batch_data_sample)
                
            if key == 'Rotate' and val:
                if isinstance(val, str):
                    val = [int(x.strip()) for x in val.split(',')]
                elif isinstance(val, int):
                    val = [val]
                
                for degree in val:                    
                    new_data_batch['inputs'].append(RandomRotation(degree)(batch_input))
                    new_data_batch['data_samples'].append(batch_data_sample)
                    
            if key == 'Translate' and val:
                if isinstance(val, str):
                    val = [int(x.strip()) for x in val.split(',')]
                elif isinstance(val, int):
                    val = [val]
                    
                new_data_batch['inputs'].append(translate_tensor(batch_input, val[0], val[1]))
                new_data_batch['data_samples'].append(batch_data_sample)

    def merge_tta_batch(self, batch_output):
        seg_logits = batch_output[0].seg_logits.data
        logits = torch.zeros(seg_logits.shape).to(seg_logits)
        for data_sample in batch_output:
            seg_logit = data_sample.seg_logits.data
            if self._model.out_channels > 1:
                logits += seg_logit.softmax(dim=0)
            else:
                logits += seg_logit.sigmoid()
        logits /= len(batch_output)
        if self._model.out_channels == 1:
            seg_pred = (logits > self._model.decode_head_threshold).to(logits).squeeze(1)
        else:
            seg_pred = logits.argmax(dim=0)
        data_sample.set_data({'seg_logits': PixelData(data=logits)})
        data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})
        if hasattr(batch_output[0], 'gt_sem_seg'):
            data_sample.set_data(
                {'gt_sem_seg': batch_output[0].gt_sem_seg})
        data_sample.set_metainfo({'img_path': batch_output[0].img_path})
        
        return data_sample
                
    def recover_tta_output(self, batch_output):
        idx = 0
        for key, val in self._augs.items():
            if key == 'HorizontalFlip' and val:
                batch_output[idx + 1].seg_logits = PixelData(data=RandomHorizontalFlip(1)(batch_output[idx + 1].seg_logits.data))
                idx += 1
        
            if key == 'VerticalFlip' and val:
                batch_output[idx + 1].seg_logits = PixelData(data=RandomVerticalFlip(1)(batch_output[idx + 1].seg_logits.data))
                idx += 1
                
            if key == 'Rotate' and val:
                if isinstance(val, str):
                    val = [int(x.strip()) for x in val.split(',')]
                elif isinstance(val, int):
                    val = [val]
                
                for degree in val:                    
                    batch_output[idx + 1].seg_logits = PixelData(data=RandomRotation(360 - degree)(batch_output[idx + 1].seg_logits.data))
                    idx += 1
                    
            if key == 'Translate' and val:
                if isinstance(val, str):
                    val = [int(x.strip()) for x in val.split(',')]
                elif isinstance(val, int):
                    val = [val]
                
                batch_output[idx + 1].seg_logits = PixelData(data=translate_tensor(batch_output[idx + 1].seg_logits.data, -val[0], -val[1]))
                idx += 1
                
