import os
import os.path as osp
import imgviz
import numpy as np
import cv2
import random


def vis_val(outputs, ratio, output_dir, current_epoch):
    
    color_map = imgviz.label_colormap(50)
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    output_dir = osp.join(output_dir, str(current_epoch))
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    for output in outputs:
        if random.random() <= ratio:
            img_path = output.img_path 
            filename = osp.split(osp.splitext(img_path)[0])[-1]
            img = cv2.imread(img_path)
            seg_logits = output.seg_logits.data.cpu().detach().numpy()
            gt_sem_seg = output.gt_sem_seg.data.squeeze(0).cpu().detach().numpy()
            pred_sem_seg = output.pred_sem_seg.data.squeeze(0).cpu().detach().numpy()

            height, width = output.ori_shape
            # img_shape = output.img_shape 
            reduce_zero_label = output.reduce_zero_label
            
            if reduce_zero_label:
                gt_sem_seg[gt_sem_seg == 255] = -1
                gt_sem_seg += 1
        
        
            vis_img = np.zeros((height, width*2, 3))
    
            if img.shape[:2] != gt_sem_seg.shape[:2]:
                gt_sem_seg = cv2.resize(gt_sem_seg.astype(np.uint8), (_input.shape[1], _input.shape[0]))
                
            vis_img[:, :width, :] = cv2.addWeighted(img, 0.4, color_map[gt_sem_seg], 0.6, 0)
            vis_img[:, width:, :] = cv2.addWeighted(img, 0.4, color_map[pred_sem_seg], 0.6, 0)
            cv2.imwrite(osp.join(output_dir, filename + '.png'), vis_img)
            
            
        