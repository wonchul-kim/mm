from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from mmengine.registry import LOOPS
from mmengine.runner.amp import autocast
from mmengine.runner.loops import TestLoop, _parse_losses, _update_losses

def translate_tensor(tensor, shift_x, shift_y):
    import torch.nn.functional as F
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

@LOOPS.register_module()
class TestLoopV2(TestLoop):
    def run(self) -> dict:
        """Launch test."""
        self.runner.call_hook('before_test')
        self.runner.call_hook('before_test_epoch')
        self.runner.model.eval()

        self.test_loss.clear()
        for idx, data_batch in enumerate(self.dataloader):
            if hasattr(data_batch['data_samples'][0], 'patch') and data_batch['data_samples'][0].patch:
                _size = self.run_iter_patch(idx, data_batch)
                metrics = self.evaluator.evaluate(_size)
            else:
                if hasattr(self.runner.cfg, 'tta') and self.runner.cfg.tta['use']:
                    self.run_iter_tta(idx, data_batch)
                else:
                    self.run_iter(idx, data_batch)
                metrics = self.evaluator.evaluate(len(self.dataloader.dataset))

        if self.test_loss:
            loss_dict = _parse_losses(self.test_loss, 'test')
            metrics.update(loss_dict)

        self.runner.call_hook('after_test_epoch', metrics=metrics)
        self.runner.call_hook('after_test')
        return metrics

    @torch.no_grad()
    def run_iter_patch(self, idx, data_batch: Sequence[dict]) -> None:
        import copy
        import numpy as np
        import os
        import os.path as osp
        import cv2
        import imgviz
        from mmengine.structures import PixelData
        
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
        
        for custom_hook in self.runner.cfg['custom_hooks']:
            if 'type' in custom_hook and custom_hook['type'] == 'VisualizeTest':
                output_dir = custom_hook['output_dir']
                annotate = custom_hook['annotate']
                contour_thres = custom_hook['contour_thres']
                
        if not osp.exists(output_dir):
            os.mkdir(output_dir)    
        
        color_map = imgviz.label_colormap(50)
        inputs, data_samples = [], []
        batch_size = len(data_batch['inputs'])
        eval_cnt = 0
        for input_image, data_sample in zip(data_batch['inputs'], data_batch['data_samples']):
                       
            patch_info = data_sample.patch
            roi = data_sample.roi
            filename = osp.split(osp.splitext(data_sample.img_path)[0])[-1]
            
            if len(roi) == 0:
                roi = [0, 0, data_sample.img_shape[1], data_sample.img_shape[0]]
                
            dx = int((1.0 - patch_info['overlap_ratio']) * patch_info['width'])
            dy = int((1.0 - patch_info['overlap_ratio']) * patch_info['height'])

            vis_gt, vis_pred = np.zeros((roi[3] - roi[1], roi[2] - roi[0], 3)), np.zeros((roi[3] - roi[1], roi[2] - roi[0], 3))

            # annotate
            if annotate or not osp.exists(data_sample.seg_map_path):
                from visionsuite.utils.dataset.formats.labelme.utils import get_points_from_image, init_labelme_json

                annotation_dir = osp.join(output_dir, '..', 'labels')
                if not osp.exists(annotation_dir):
                    os.makedirs(annotation_dir, exist_ok=True)
                
                _labelme = init_labelme_json(filename + ".bmp", roi[2] - roi[0], roi[3] - roi[1])
                
            else:
                _labelme = None
            
            # patch inference
            for y0 in range(roi[1], roi[3], dy):
                for x0 in range(roi[0], roi[2], dx):
                    
                    patch_data_sample = copy.deepcopy(data_sample)
                    patch_input_image = copy.deepcopy(input_image)
                    if y0 + patch_info['height'] > roi[3] - roi[1]:
                        y = roi[3] - roi[1] - patch_info['height']
                    else:
                        y = y0

                    if x0 + patch_info['width'] > roi[2] - roi[0]:
                        x = roi[2] - roi[0] - patch_info['width']
                    else:
                        x = x0
                        
                    patch_input_image = patch_input_image[:, y:y + patch_info['height'], x:x + patch_info['width']]
                    
                    patch_data_sample.gt_sem_seg = PixelData(data=data_sample.gt_sem_seg.data[:, y:y + patch_info['height'], x:x + patch_info['width']])
                    patch_data_sample.set_metainfo({'patch': [x, y, x + patch_info['width'], y + patch_info['height']]})
                    patch_data_sample.set_metainfo({'ori_shape': (patch_info['height'], patch_info['width'])})
                    patch_data_sample.set_metainfo({'img_shape': (patch_info['height'], patch_info['width'])})
                    
                    inputs.append(patch_input_image)
                    data_samples.append(patch_data_sample)
                    
                    if batch_size%len(inputs) == 0:
                        patch_data_batch = {'inputs': inputs, 'data_samples': data_samples}   
                    
                        with autocast(enabled=self.fp16):
                            if hasattr(self.runner.cfg, 'tta') and self.runner.cfg.tta['use']:
                                outputs = self.run_tta_batch(patch_data_batch)
                            else:
                                outputs = self.runner.model.test_step(patch_data_batch)

                        outputs, self.test_loss = _update_losses(outputs, self.test_loss)
                        eval_outputs, eval_inputs, eval_data_samples = [], [], []
                        for jdx, output in enumerate(outputs):
                            if osp.exists(output.seg_map_path):
                                eval_outputs.append(output)
                                eval_inputs.append(patch_data_batch['inputs'][jdx])
                                eval_data_samples.append(patch_data_batch['data_samples'][jdx])
                                eval_cnt += 1
                                
                            if _labelme:
                                _labelme = get_points_from_image(output.pred_sem_seg.data.squeeze(0).cpu().detach().numpy(), 
                                                                 output.classes,
                                                                 roi,
                                                                 [x, y],
                                                                 _labelme,
                                                                 contour_thres,
                                                            )
                        self.evaluator.process(data_samples=eval_outputs, 
                                                       data_batch={'inputs': eval_inputs, 
                                                                   'data_samples': eval_data_samples}
                                                )
                        
                                
                        self.runner.call_hook(
                                'after_test_iter',
                                batch_idx=idx,
                                data_batch=data_batch,
                                outputs=outputs)
                            
                        for _input, output in zip(inputs, outputs):
                            vis_gt[output.patch[1]:output.patch[3], output.patch[0]:output.patch[2]] = cv2.addWeighted(np.transpose(_input.cpu().detach().numpy(), (1, 2, 0)), 0.4, color_map[output.gt_sem_seg.data.squeeze(0).cpu().detach().numpy()], 0.6, 0)
                            vis_pred[output.patch[1]:output.patch[3], output.patch[0]:output.patch[2]] = cv2.addWeighted(np.transpose(_input.cpu().detach().numpy(), (1, 2, 0)), 0.4, color_map[output.pred_sem_seg.data.squeeze(0).cpu().detach().numpy()], 0.6, 0)
                        
                        inputs, data_samples = [], []
                        
            cv2.rectangle(vis_gt, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
            cv2.rectangle(vis_pred, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
            cv2.imwrite(osp.join(output_dir, filename + '.png'), np.hstack((vis_gt, vis_pred)))
            
            if _labelme:
                import json
                
                with open(os.path.join(annotation_dir, filename + ".json"), "w") as jsf:
                    json.dump(_labelme, jsf)
                


        return eval_cnt
    
    @torch.no_grad()
    def run_iter_tta(self, idx, data_batch: Sequence[dict]) -> None:
        """Iterate one mini-batch.

        Args:
            data_batch (Sequence[dict]): Batch of data from dataloader.
        """
        self.runner.call_hook(
            'before_test_iter', batch_idx=idx, data_batch=data_batch)
                
        with autocast(enabled=self.fp16):
            outputs = self.run_tta_batch(data_batch)

        outputs, self.test_loss = _update_losses(outputs, self.test_loss)

        self.evaluator.process(data_samples=outputs, data_batch=data_batch)
        self.runner.call_hook(
            'after_test_iter',
            batch_idx=idx,
            data_batch=data_batch,
            outputs=outputs)
    

                           
                            
    @torch.no_grad()
    def run_tta_batch(self, data_batch):               
        from torchvision.transforms import RandomHorizontalFlip, RandomVerticalFlip, RandomRotation
        from mmengine.structures import PixelData

        augs = self.runner.cfg.tta['augs']
        batch_inputs = data_batch['inputs']
        batch_data_samples = data_batch['data_samples']
        
        new_data_batch = {'inputs': [], 'data_samples': []}
        batch_outputs = []
        for batch_input, batch_data_sample in zip(batch_inputs, batch_data_samples):
            new_data_batch['inputs'].append(batch_input)
            new_data_batch['data_samples'].append(batch_data_sample)
            for key, val in augs.items():
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
                    
            batch_output = self.runner.model.test_step(new_data_batch)
            
            idx = 0
            for key, val in augs.items():
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
            
            seg_logits = batch_output[0].seg_logits.data
            logits = torch.zeros(seg_logits.shape).to(seg_logits)
            for data_sample in batch_output:
                seg_logit = data_sample.seg_logits.data
                if self.runner.model.out_channels > 1:
                    logits += seg_logit.softmax(dim=0)
                else:
                    logits += seg_logit.sigmoid()
            logits /= len(batch_output)
            if self.runner.model.out_channels == 1:
                seg_pred = (logits > self.runner.model.decode_head.threshold
                            ).to(logits).squeeze(1)
            else:
                seg_pred = logits.argmax(dim=0)
            data_sample.set_data({'seg_logits': PixelData(data=logits)})
            data_sample.set_data({'pred_sem_seg': PixelData(data=seg_pred)})
            if hasattr(batch_output[0], 'gt_sem_seg'):
                data_sample.set_data(
                    {'gt_sem_seg': batch_output[0].gt_sem_seg})
            data_sample.set_metainfo({'img_path': batch_output[0].img_path})
            batch_outputs.append(data_sample)
            
        return batch_outputs