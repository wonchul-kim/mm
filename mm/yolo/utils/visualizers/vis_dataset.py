import cv2
import random
import numpy as np
import imgviz
import os
import os.path as osp
import torch


def vis_dataset(dataset, mode, ratio=0.5, output_dir='/HDD/etc/outputs/mm'):
    color_map = imgviz.label_colormap(50)[1:]
    cnt = 0
    
    output_dir = osp.join(output_dir, mode)
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    for batch in dataset:
        if random.random() <= ratio:
            _input = batch['inputs'] 
            data_samples = batch['data_samples'] 
            # including ori_shape, reduce_zero_label, img_path, seg_map_path, scale_factor, img_shape, gt_sem_seg(PixelData)
            assert isinstance(_input, torch.Tensor), RuntimeError(f"To debug dataset, the input must be tensor")
            cnt += 1
            _input = _input.permute(1, 2, 0).cpu().detach().numpy()
            _input = np.ascontiguousarray(_input)
            filename = osp.split(osp.splitext(data_samples.img_path)[0])[-1]
            ignored_instances = data_samples.ignored_instances 
            gt_instances = data_samples.gt_instances 
            
            bboxes = gt_instances.bboxes.cpu().detach().numpy()
            labels = gt_instances.labels.cpu().detach().numpy()
            for bbox, label in zip(bboxes, labels):
                cv2.rectangle(_input, tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:])), 
                            color=tuple(map(int, color_map[label])), thickness=2)
                cv2.putText(_input, str(label), (int(bbox[0] - 10), int(bbox[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(map(int, color_map[label])), 2)

            cv2.imwrite(osp.join(output_dir, filename + '.png'), _input)
    
                