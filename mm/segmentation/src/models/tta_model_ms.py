import torch
import torch.nn.functional as F
from typing import Optional

from mmengine.structures import PixelData
from mmseg.structures import SegDataSample

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

def random_horizontal_flip(tensor):
    return torch.flip(tensor, dims=[-1])


def random_vertical_flip(tensor):
    return torch.flip(tensor, dims=[-2])

def multiscale_tensor(tensor, ratio):
    original_w, original_h = tensor.shape[2], tensor.shape[1]
    resized_tensor = F.interpolate(tensor.unsqueeze(0), size=(int(original_h*ratio), int(original_w*ratio)), 
                                   mode='bilinear', align_corners=False)
    
    return resized_tensor[0]
    
    

class TTASegModel(torch.nn.Module):
    def __init__(self, model, augs, shape=None):
        super().__init__()
        self._model = model 
        self._augs = augs
        self._shape = shape
    
    def forward(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor'):
        """
        Returns:
            outputs: logits - tensor shaped (bs, num_classes, height, width)
        """

        outputs = self.forward_tta(inputs=inputs, data_samples=data_samples, mode=mode)
        # outputs = self.model(inputs=inputs, data_samples=data_samples, mode=mode)
        
        return outputs
    
    def forward_tta(self, inputs: torch.Tensor, data_samples: Optional[list] = None, mode: str = 'tensor'):
        batch_inputs = inputs
        
        if data_samples is None:
            data_samples = [SegDataSample()]

        for data_sample in data_samples:
            data_sample.set_field(
                name='img_shape', value=self._shape, field_type='metainfo')
        
        batch_data_samples = data_samples
        batch_outputs = []
        for batch_input, batch_data_sample in zip(batch_inputs, batch_data_samples):
            new_data_batch = {'inputs': [], 'data_samples': []}
            self.make_tta_batch(new_data_batch, batch_input, batch_data_sample)
            batch_output = self.model(inputs=torch.stack(new_data_batch['inputs'], dim=0), 
                                    data_samples=new_data_batch['data_samples'], mode=mode)
            self.recover_tta_output(batch_output)
            merged_output = self.merge_tta_batch(batch_output)
            batch_outputs.append(merged_output)
            
        return torch.stack(batch_outputs, dim=0)
    
    def test_step(self, data_batch):
        return self._model.test_step(data_batch)
            
    def run_tta_batch(self, data_batch):
        batch_inputs = data_batch['inputs']
        batch_data_samples = data_batch['data_samples']
        
        batch_outputs = []
        for batch_input, batch_data_sample in zip(batch_inputs, batch_data_samples):
            new_data_batches = [{'inputs': [], 'data_samples': []}]
            self.make_tta_batch(new_data_batches, batch_input, batch_data_sample)
            for index, new_data_batch in enumerate(new_data_batches):
                batch_output = self._model.test_step(new_data_batch)
                self.recover_tta_output(batch_output, index)
                batch_outputs.append(self.merge_tta_batch(batch_output))
        return batch_outputs
            
    @property
    def model(self):
        return self._model 

    @property
    def augs(self):
        return self._augs 
    
    @property
    def shape(self):
        return self._shape 
    
    def __getattr__(self, name):
        if name in ['test_step', '_model', 'forward']:
            return super().__getattr__(name)
        else:
            return getattr(self._model, name)

    def make_tta_batch(self, new_data_batches, batch_input, batch_data_sample):
        new_data_batches[0]['inputs'].append(batch_input)
        new_data_batches[0]['data_samples'].append(batch_data_sample)
        for key, val in self._augs.items():
            if key == 'HorizontalFlip' and val:
                new_data_batches[0]['inputs'].append(random_horizontal_flip(batch_input))
                new_data_batches[0]['data_samples'].append(batch_data_sample)
            
            if key == 'VerticalFlip' and val:
                new_data_batches[0]['inputs'].append(random_vertical_flip(batch_input))
                new_data_batches[0]['data_samples'].append(batch_data_sample)
                
            # if key == 'Rotate' and val:
            #     if isinstance(val, str):
            #         val = [int(x.strip()) for x in val.split(',')]
            #     elif isinstance(val, int):
            #         val = [val]
                
            #     for degree in val:                    
            #         new_data_batches[0]['inputs'].append(RandomRotation(degree)(batch_input))
            #         new_data_batches[0]['data_samples'].append(batch_data_sample)
                    
            if key == 'Translate' and (val != '' and val != []):
                if isinstance(val, str):
                    val = [int(x.strip()) for x in val.split(',')]
                elif isinstance(val, int):
                    val = [val]
                    
                new_data_batches[0]['inputs'].append(translate_tensor(batch_input, val[0], val[1]))
                new_data_batches[0]['data_samples'].append(batch_data_sample)
                
            if key == 'Multiscale' and (val != '' and val != []):
                if isinstance(val, str):
                    val = [float(x.strip()) for x in val.split(',')]
                elif isinstance(val, float):
                    val = [val]
                
                for _val in val:
                    new_data_batch = {'inputs': [], 'data_samples': []}
                    new_data_batch['inputs'].append(multiscale_tensor(batch_input, _val))
                    new_data_batch['data_samples'].append(batch_data_sample)
                    new_data_batches.append(new_data_batch)
        
    def merge_tta_batch(self, batch_output):
        if isinstance(batch_output, torch.Tensor):
            seg_logits = batch_output[0]
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in batch_output:
                seg_logit = data_sample
                if self._model.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
            logits /= len(batch_output)
            if self._model.out_channels == 1:
                seg_pred = (logits > self._model.decode_head_threshold).to(logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)
                
            return logits
        else:
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
                
    def recover_tta_output(self, batch_output, index):
        
        if index == 0 and (self._augs['HorizontalFlip'] or self._augs['VerticalFlip'] or (self._augs['Translate'] != '' and self._augs['Translate'] != [])):
            idx = 0
            for key, val in self._augs.items():
                if key == 'HorizontalFlip' and val:
                    if isinstance(batch_output[idx + 1], torch.Tensor):
                        batch_output[idx + 1] = random_horizontal_flip(batch_output[idx + 1])
                    else:
                        batch_output[idx + 1].seg_logits = PixelData(data=random_horizontal_flip(batch_output[idx + 1].seg_logits.data))
                    idx += 1
            
                if key == 'VerticalFlip' and val:
                    if isinstance(batch_output[idx + 1], torch.Tensor):
                        batch_output[idx + 1] = random_vertical_flip(batch_output[idx + 1])
                    else:
                        batch_output[idx + 1].seg_logits = PixelData(data=random_vertical_flip(batch_output[idx + 1].seg_logits.data))
                    idx += 1
                    
                # if key == 'Rotate' and val and val != 0:
                #     if isinstance(val, str):
                #         val = [int(x.strip()) for x in val.split(',')]
                #     elif isinstance(val, int):
                #         val = [val]
                    
                #     for degree in val:      
                #         if isinstance(batch_output[idx + 1], torch.Tensor):              
                #             batch_output[idx + 1] = RandomRotation(360 - degree)(batch_output[idx + 1])
                #         else:
                #             batch_output[idx + 1].seg_logits = PixelData(data=RandomRotation(360 - degree)(batch_output[idx + 1].seg_logits.data))
                #         idx += 1
                        
                if key == 'Translate' and val:
                    if isinstance(val, str):
                        val = [int(x.strip()) for x in val.split(',')]
                    elif isinstance(val, int):
                        val = [val]
                    if isinstance(batch_output[idx + 1], torch.Tensor):
                        batch_output[idx + 1] = translate_tensor(batch_output[idx + 1], -val[0], -val[1])
                    else:
                        batch_output[idx + 1].seg_logits = PixelData(data=translate_tensor(batch_output[idx + 1].seg_logits.data, -val[0], -val[1]))
                    idx += 1

                    
if __name__ == '__main__':
    import cv2
    from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation

    # img_file = "/DeepLearning/_athena_tests/datasets/polygon2/split_dataset/train/0_124062721060032_6_Outer.bmp"
    img_file = "/DeepLearning/_athena_tests/datasets/rectangle1/split_dataset/val/14_122083110333055_3_Socket_Bottom.bmp"
    img = cv2.imread(img_file)
    
    input_tensor = torch.from_numpy(img)
    # flipped_tensor = random_horizontal_flip(input_tensor)
    flipped_tensor = RandomHorizontalFlip(1)(input_tensor)
    cv2.imwrite("/HDD/etc/etc/hflip_image.bmp", flipped_tensor.numpy())
    # flipped_tensor = random_vertical_flip(input_tensor)
    # cv2.imwrite("/HDD/etc/etc/vflip_image.bmp", flipped_image)
