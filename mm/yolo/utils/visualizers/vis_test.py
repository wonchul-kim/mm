import cv2
import numpy as np
import imgviz
import os
import os.path as osp
import torch
import copy
import json
from visionsuite.utils.dataset.formats.labelme.utils import add_labelme_element, init_labelme_json
from mm.utils.functions import numpy_converter

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

def vis_test(outputs, data_batch, output_dir, batch_idx,
             annotate=False, conf_threshold=0.2, save_raw=False, classes=None,
             color_map = imgviz.label_colormap(50)[1:], bbox_thickness=2):
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    if classes is None:
        label2class = None
    else:
        label2class = {label: _class for label, _class in enumerate(classes)}
        
    for _input, output in zip(data_batch['inputs'], outputs):
        assert isinstance(_input, torch.Tensor), RuntimeError(f"To debug dataset, the input must be tensor")
        
        filename = osp.split(osp.splitext(output.img_path)[0])[-1]
        gt_instances = output.gt_instances 
        pred_instances = output.pred_instances 
        scale_factor = output.scale_factor # [0.3636, 0.362990]
        pad_param = list(map(int, output.pad_param)) # [169, 169, 0, 0]
        ori_shape = list(map(int, output.ori_shape))
        img_shape = output.img_shape
        
        # to annotate
        if annotate:
            annotate_dir = osp.join(output_dir, '../labels')
            if not osp.exists(annotate_dir):
                os.mkdir(annotate_dir)
            _labelme = init_labelme_json(filename + ".bmp", ori_shape[1], ori_shape[0])
        
        _input = _input.permute(1, 2, 0).cpu().detach().numpy()
        _input = np.ascontiguousarray(_input)
        _input = restore_image(_input, pad_param, ori_shape)
        pred_input = copy.deepcopy(_input)
        
        pred_bboxes = pred_instances.bboxes.cpu().detach().numpy()
        pred_labels = pred_instances.labels.cpu().detach().numpy()
        pred_scores = pred_instances.scores.cpu().detach().numpy()
        
        for bbox, label, score in zip(pred_bboxes, pred_labels, pred_scores):
            if score >= conf_threshold:
                cv2.rectangle(pred_input, tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:])), 
                                color=tuple(map(int, color_map[label])), thickness=bbox_thickness)
                cv2.putText(pred_input, f"{label}_{score:.2f}" if label2class is None else f"{label2class[label]}_{score:.2f}", 
                            (int(bbox[0] - 10), int(bbox[1] - 10)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(map(int, color_map[label])), 2)
                
                if annotate:
                    points, point = [], []
                    for bbox_idx, val in enumerate(bbox):
                        point.append(val)
                        
                        if bbox_idx%2 == 1:
                            points.append(point)
                            point = []
                        
                    _labelme = add_labelme_element(
                                                    _labelme,
                                                    shape_type='rectangle',
                                                    label=int(label) if label2class is None else label2class[label],
                                                    points=points,
                                                    otherData={"confidence": score},
                                                )
                    
        with open(os.path.join(annotate_dir, filename + ".json"), "w") as json_file:
            json.dump(_labelme, json_file, default=numpy_converter)
            
        gt_bboxes = gt_instances.bboxes.cpu().detach().numpy()
        gt_labels = gt_instances.labels.cpu().detach().numpy()
        for bbox, label in zip(gt_bboxes, gt_labels):
            cv2.rectangle(_input, tuple(map(int, bbox[:2])), tuple(map(int, bbox[2:])), 
                            color=tuple(map(int, color_map[label])), thickness=2)
            cv2.putText(_input, f"{label}", (int(bbox[0] - 10), int(bbox[1] - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, tuple(map(int, color_map[label])), 2)

        cv2.imwrite(osp.join(output_dir, filename + '.png'), np.hstack([_input, pred_input]))
        