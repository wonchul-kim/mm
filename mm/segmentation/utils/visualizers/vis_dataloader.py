import cv2
import random
import numpy as np
import imgviz
import os
import os.path as osp
import torch


def vis_dataloader(dataloader, mode, ratio=0.5, output_dir='/HDD/etc/outputs/mm'):
    color_map = imgviz.label_colormap(50)
    cnt, done = 0, False
    
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    output_dir = osp.join(output_dir, mode)
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for batch in dataloader:
        if random.random() <= ratio:
            inputs = batch['inputs'] # list of tensors
            data_samples = batch['data_samples'] # SegDataSample w/ meta
            # including ori_shape, reduce_zero_label, img_path, seg_map_path, scale_factor, img_shape, gt_sem_seg(PixelData)
            if isinstance(inputs, list):
                for _input, data_sample in zip(inputs, data_samples):
                    assert isinstance(_input, torch.Tensor), RuntimeError(f"To debug dataset, the input must be tensor")
                    cnt += 1
                    _input = _input.permute(1, 2, 0).cpu().detach().numpy()
                    height, width, channel = _input.shape
                    img_path = data_sample.img_path 
                    reduce_zero_label = data_sample.reduce_zero_label
                    filename = osp.split(osp.splitext(img_path)[0])[-1]
                    seg_map_path = data_sample.seg_map_path 
                    # scale_factor = data_sample.scale_factor 
                    img_shape = data_sample.img_shape 
                    ori_shape = data_sample.ori_shape 
                    gt_sem_seg = data_sample.gt_sem_seg.data.squeeze(0).cpu().detach().numpy()
                    
                    if reduce_zero_label:
                        gt_sem_seg[gt_sem_seg == 255] = -1
                        gt_sem_seg += 1
                    
                    vis_img = np.zeros((height, width*2, channel))
                    vis_img[:, :width, :] = _input 
                    
                    if _input.shape[:2] != gt_sem_seg.shape[:2]:
                        gt_sem_seg = cv2.resize(gt_sem_seg.astype(np.uint8), (_input.shape[1], _input.shape[0]))
                    
                    vis_img[:, width:, :] = cv2.addWeighted(_input, 0.4, color_map[gt_sem_seg], 0.6, 0)
                    cv2.imwrite(osp.join(output_dir, filename + '.png'), vis_img)
                    
                    if cnt >= len(dataloader)*ratio:
                        done = True 
                        break
                    
                if done:
                    break
            else:
                assert isinstance(inputs, torch.Tensor), RuntimeError(f"To debug dataset, the input must be tensor")
                cnt += 1
                inputs = inputs.permute(1, 2, 0).cpu().detach().numpy()
                height, width, channel = inputs.shape
                img_path = data_samples.img_path 
                reduce_zero_label = data_samples.reduce_zero_label
                filename = osp.split(osp.splitext(img_path)[0])[-1]
                seg_map_path = data_samples.seg_map_path 
                # scale_factor = data_samples.scale_factor 
                img_shape = data_samples.img_shape 
                ori_shape = data_samples.ori_shape 
                gt_sem_seg = data_samples.gt_sem_seg.data.squeeze(0).cpu().detach().numpy()
                
                if reduce_zero_label:
                    gt_sem_seg[gt_sem_seg == 255] = -1
                    gt_sem_seg += 1
                
                vis_img = np.zeros((height, width*2, channel))
                vis_img[:, :width, :] = inputs 
                
                if inputs.shape[:2] != gt_sem_seg.shape[:2]:
                    gt_sem_seg = cv2.resize(gt_sem_seg.astype(np.uint8), (inputs.shape[1], inputs.shape[0]))
                
                vis_img[:, width:, :] = cv2.addWeighted(inputs, 0.4, color_map[gt_sem_seg], 0.6, 0)
                cv2.imwrite(osp.join(output_dir, filename + '.png'), vis_img)
                
                if cnt >= len(dataloader)*ratio:
                    done = True 
                    break