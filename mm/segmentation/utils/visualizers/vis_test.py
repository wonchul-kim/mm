import os
import os.path as osp
import imgviz
import numpy as np
import cv2

from visionsuite.utils.dataset.formats.labelme.utils import get_points_from_image, init_labelme_json
from visionsuite.utils.helpers import get_text_coords

def vis_test(outputs, output_dir, data_batch, batch_idx, annotate=False, 
             contour_thres=10, contour_conf=0.5, create_parent_path=False,
             save_raw=False, legend=True):
    
    if not (hasattr(outputs[0], 'patch') and len(outputs[0].patch) != 0):
        color_map = imgviz.label_colormap(50)
        if not osp.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            
        if save_raw:
            raw_dir = osp.join(output_dir, '../raw')
            if not osp.exists(raw_dir):
                os.mkdir(raw_dir)
            
        logits_dir = osp.join(output_dir, '../logits')
        if not osp.exists(logits_dir):
            os.makedirs(logits_dir, exist_ok=True)
            
        for jdx, (input_image, output) in enumerate(zip(data_batch['inputs'], outputs)):
            # img
            img_path = output.img_path 
            filename = osp.split(osp.splitext(img_path)[0])[-1]
            parent_path = None
            if output.is_parent_path:
                parent_path = osp.split(osp.splitext(img_path)[0])[-2]
                parent_path = '/'.join(parent_path.split("/")[-output.is_parent_path:])        
            
            # output
            classes = output.classes[1:]
            seg_logits = output.seg_logits.data.cpu().detach().numpy()
            gt_sem_seg = output.gt_sem_seg.data.squeeze(0).cpu().detach().numpy()
            pred_sem_seg = output.pred_sem_seg.data.squeeze(0).cpu().detach().numpy()
            reduce_zero_label = output.reduce_zero_label
            ori_shape = output.ori_shape # (h, w)
            
            # raw
            if save_raw:
                np.save(osp.join(raw_dir, filename + f'_{batch_idx}_{jdx}.npy'), seg_logits)

            # roi
            if len(output.roi) == 0:
                roi = [0, 0, ori_shape[1], ori_shape[0]]
            else:
                roi = output.roi
                
            # resize
            if ori_shape != output.img_shape:
                input_image = cv2.imread(output.img_path)
                input_image = input_image[roi[1]:roi[3], roi[0]:roi[2], :]
            else:
                input_image = np.transpose(input_image.cpu().detach().numpy(), (1, 2, 0))
            input_height, input_width = input_image.shape[:2]
                
            if osp.exists(output.seg_map_path):
                gt_vis_img = input_image.copy()
                gt_vis_img = cv2.addWeighted(input_image, 0.4, color_map[gt_sem_seg], 0.6, 0)
            else:
                gt_vis_img = None

            pred_vis_img = input_image.copy()
            # pred_vis_img[roi[1]:roi[3], roi[0]:roi[2]] = cv2.addWeighted(input_image[roi[1]:roi[3], roi[0]:roi[2]], 0.4, color_map[pred_sem_seg], 0.6, 0)
            pred_vis_img = cv2.addWeighted(input_image, 0.4, color_map[pred_sem_seg], 0.6, 0)
            # cv2.rectangle(pred_vis_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)

            # annotate
            if annotate or not osp.exists(output.seg_map_path):
                import json 
                
                annotation_dir = osp.join(output_dir, '..', 'labels')
                if not osp.exists(annotation_dir):
                    os.makedirs(annotation_dir, exist_ok=True)
                
                _labelme = init_labelme_json(filename + ".bmp", input_width, input_height)
                _labelme, label_points = get_points_from_image(pred_sem_seg, classes,
                                    roi,
                                    [0, 0],
                                    _labelme,
                                    contour_thres,
                                    conf=contour_conf,
                                    ret_points=True,
                                )
                
                for idx, (label, points) in enumerate(label_points.items()):
                    for point in points['bbox']:
                        font_scale = 1
                        offset_h = 10
                        cv2.putText(pred_vis_img, label, get_text_coords([[point[0] - roi[0], point[1] - roi[1]], [point[2] - roi[0], point[3] - roi[1]]], 
                                                                       input_width, input_height, offset_h=offset_h), 
                                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, tuple(map(int, color_map[idx + 1])), 3)
                
                if parent_path:
                    if create_parent_path:
                        if not osp.exists(osp.join(annotation_dir, parent_path)):
                            os.makedirs(osp.join(annotation_dir, parent_path))
                    
                        with open(os.path.join(annotation_dir, parent_path, filename + ".json"), "w") as jsf:
                            json.dump(_labelme, jsf)
                    else:
                        with open(os.path.join(annotation_dir, '_'.join(parent_path.split('/')).replace('._', '') + '_' + filename + ".json"), "w") as jsf:
                            json.dump(_labelme, jsf)
                else:
                    with open(os.path.join(annotation_dir, filename + ".json"), "w") as jsf:
                        json.dump(_labelme, jsf)
            else:
                _labelme = None
                
            if reduce_zero_label:
                gt_sem_seg[gt_sem_seg == 255] = -1
                gt_sem_seg += 1

            if legend:
                vis_legend = np.zeros((input_height, 300, 3), dtype="uint8")
                for idx, _class in enumerate(('background', ) + classes):
                    color = [int(c) for c in color_map[idx]]
                    cv2.putText(vis_legend, _class, (5, (idx * 25) + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    cv2.rectangle(vis_legend, (150, (idx * 25)), (300, (idx * 25) + 25), tuple(color), -1)
                # vis_legend = cv2.vconcat([text_legends, vis_legend])
            else:
                vis_legend = None

            if gt_vis_img is not None:
                if vis_legend is not None:
                    vis_img = np.hstack((gt_vis_img, pred_vis_img, vis_legend))
                else:
                    vis_img = np.hstack((gt_vis_img, pred_vis_img))
            else:
                if vis_legend is not None:
                    vis_img = np.hstack((pred_vis_img, vis_legend))
                else:
                    vis_img = pred_vis_img
                
            if parent_path:
                if create_parent_path:
                    if not osp.exists(osp.join(output_dir, parent_path)):
                        os.makedirs(osp.join(output_dir, parent_path))
                    cv2.imwrite(osp.join(output_dir, parent_path, filename + f'_{batch_idx}_{jdx}.png'), vis_img)
                else:
                    cv2.imwrite(osp.join(output_dir, '_'.join(parent_path.split('/')).replace('._', '') + '_' + filename + f'_{batch_idx}_{jdx}.png'), vis_img)
            else:
                cv2.imwrite(osp.join(output_dir, filename + f'_{batch_idx}_{jdx}.png'), vis_img)
                
            heatmaps = []
            for logit_idx, (seg_logit, class_name) in enumerate(zip(seg_logits, classes)):
                origin = 100, 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_img_height = 50
                txt_img = np.zeros((txt_img_height, input_width, 3), np.uint8)
                cv2.putText(txt_img, class_name, origin, font, 1, (255, 255, 255), 1)
                heatmap = cv2.normalize(seg_logit, None, 0, 255, cv2.NORM_MINMAX) 
                heatmap = np.uint8(heatmap)  
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.vconcat([txt_img, heatmap])
                heatmaps.append(heatmap)

            colorbar = np.linspace(0, 255, input_height + txt_img_height, dtype=np.uint8).reshape(input_height + txt_img_height, 1)
            colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET) 
            colorbar = cv2.resize(colorbar, (20, input_height + txt_img_height)) 
            
            if gt_vis_img is not None:
                origin = 100, 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_img_height = 50
                txt_img = np.zeros((txt_img_height, input_width, 3), np.uint8)
                cv2.putText(txt_img, "GT", origin, font, 1, (255, 255, 255), 1)
                gt_vis_heatmap = cv2.vconcat([txt_img, gt_vis_img])
                vis_heatmap = cv2.hconcat([gt_vis_heatmap] + heatmaps + [colorbar])
            else:
                vis_heatmap = cv2.hconcat(heatmaps + [colorbar])
                
            if parent_path:
                if create_parent_path:
                    if not osp.exists(osp.join(logits_dir, parent_path)):
                        os.makedirs(osp.join(logits_dir, parent_path))
                    cv2.imwrite(osp.join(logits_dir, parent_path, filename + f'_{idx}_{jdx}.png'), vis_heatmap)   
                else:
                    cv2.imwrite(osp.join(logits_dir, '_'.join(parent_path.split('/')).replace('._', '') + '_' + filename + f'_{idx}_{jdx}.png'), vis_heatmap)
            else:
                cv2.imwrite(osp.join(logits_dir, filename + f'_{idx}_{jdx}.png'), vis_heatmap)
            
            