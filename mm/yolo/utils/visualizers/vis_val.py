import cv2
import random
import numpy as np
import imgviz
import os
import os.path as osp
import torch
import copy

def restore_image(_input, pad_param, ori_shape):
    # 패딩 제거
    if pad_param[0] > 0:  # 상단 패딩
        _input = _input[pad_param[0]:, :, :]
    if pad_param[1] > 0:  # 하단 패딩
        _input = _input[:-pad_param[1], :, :]
    if pad_param[2] > 0:  # 좌측 패딩
        _input = _input[:, pad_param[2]:, :]
    if pad_param[3] > 0:  # 우측 패딩
        _input = _input[:, :-pad_param[3], :]
    
    # 원본 크기로 리사이즈
    _input = cv2.resize(_input, (ori_shape[1], ori_shape[0]))
    
    return _input

def vis_val(outputs, data_batch, ratio, output_dir, current_epoch, idx):
    output_dir = osp.join(output_dir, str(current_epoch))
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    color_map = imgviz.label_colormap(50)[1:]
    cnt = 0
    
    for _input, output in zip(data_batch['inputs'], outputs):
        if random.random() <= ratio:
            assert isinstance(_input, torch.Tensor), RuntimeError(f"To debug dataset, the input must be tensor")
            cnt += 1
            
            filename = osp.split(osp.splitext(output.img_path)[0])[-1]
            gt_instances = output.gt_instances 
            pred_instances = output.pred_instances 
            scale_factor = output.scale_factor # [0.3636, 0.362990]
            pad_param = list(map(int, output.pad_param)) # [169, 169, 0, 0]
            ori_shape = list(map(int, output.ori_shape))
            img_shape = output.img_shape
            
            _input = _input.permute(1, 2, 0).cpu().detach().numpy()
            _input = np.ascontiguousarray(_input)
            _input = restore_image(_input, pad_param, ori_shape)
            pred_input = copy.deepcopy(_input)
            
            pred_bboxes = pred_instances.bboxes.cpu().detach().numpy()
            pred_labels = pred_instances.labels.cpu().detach().numpy()
            pred_scores = pred_instances.scores.cpu().detach().numpy()
            for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
                cv2.rectangle(pred_input, tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:])), 
                              color=tuple(map(int, color_map[label])), thickness=2)
                cv2.putText(pred_input, f"{label}_{score}", (int(bbox[0] - 10), int(bbox[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(map(int, color_map[label])), 2)
                
            gt_bboxes = gt_instances.bboxes.cpu().detach().numpy()
            gt_labels = gt_instances.labels.cpu().detach().numpy()
            for bbox, label in zip(gt_bboxes, gt_labels):
                cv2.rectangle(_input, tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:])), 
                              color=tuple(map(int, color_map[label])), thickness=2)
                cv2.putText(_input, f"{label}", (int(bbox[0] - 10), int(bbox[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(map(int, color_map[label])), 2)

            cv2.imwrite(osp.join(output_dir, filename + '.png'), np.hstack([_input, pred_input]))
            