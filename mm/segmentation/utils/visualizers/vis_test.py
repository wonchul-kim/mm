import os
import os.path as osp
import imgviz
import numpy as np
import cv2
import torch

from visionsuite.utils.dataset.formats.labelme.utils import get_points_from_image, init_labelme_json
from visionsuite.utils.helpers import get_text_coords


# def refined_argmax(logits, threshold=0.2, bg_class=0):
#     """
#     logits: np.ndarray of shape (C, H, W)
#     threshold: float
#     Returns: np.ndarray of shape (H, W) with class indices
#     """
#     C, H, W = logits.shape

#     # softmax over class dimension
#     probs = np.exp(logits - np.max(logits, axis=0, keepdims=True))  # for numerical stability
#     probs /= np.sum(probs, axis=0, keepdims=True)

#     # top2 indices and their values
#     top2_idx = np.argsort(probs, axis=0)[-2:]  # (2, H, W)
#     top2_val = np.take_along_axis(probs, top2_idx, axis=0)  # (2, H, W)

#     result = np.full((H, W), bg_class, dtype=np.uint8)

#     for y in range(H):
#         for x in range(W):
#             idx1, idx2 = top2_idx[:, y, x]
#             val1, val2 = top2_val[:, y, x]

#             # ensure idx1 has higher prob
#             if val1 < val2:
#                 idx1, idx2 = idx2, idx1
#                 val1, val2 = val2, val1

#             if bg_class in (idx1, idx2):
#                 # one of them is background
#                 other_idx = idx1 if idx2 == bg_class else idx2
#                 other_val = val1 if idx2 == bg_class else val2
#                 if other_val > threshold:
#                     result[y, x] = other_idx
#                 else:
#                     result[y, x] = bg_class
#             else:
#                 # both are non-bg
#                 if val1 > threshold and val2 > threshold:
#                     result[y, x] = idx1  # higher one
#                 elif val1 > threshold:
#                     result[y, x] = idx1
#                 elif val2 > threshold:
#                     result[y, x] = idx2
#                 else:
#                     result[y, x] = bg_class
#     return result


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


            # def softmax(x, axis=0):
            #     x_max = np.max(x, axis=axis, keepdims=True)
            #     e_x = np.exp(x - x_max)
            #     return e_x / np.sum(e_x, axis=axis, keepdims=True)

            # threshold = 0.2
            # logits = seg_logits.copy()
            # logits[0] *= 0.3  # background penalty

            # # softmax
            # probs = softmax(logits, axis=0)  # shape: (C, H, W)

            # # reshape to (C, H*W) for easier top-k
            # C, H, W = probs.shape
            # probs_flat = probs.reshape(C, -1)  # shape: (C, H*W)

            # # top-2 indices and values along channel axis
            # top2_idx = np.argpartition(-probs_flat, kth=2, axis=0)[:2, :]  # shape: (2, H*W)
            # top2_vals = np.take_along_axis(probs_flat, top2_idx, axis=0)  # shape: (2, H*W)

            # # conditions
            # is_bg_top2 = (top2_idx == 0).sum(axis=0) == 1  # exactly one is background
            # non_bg_vals = np.where(top2_idx != 0, top2_vals, 0)  # remove bg values
            # non_bg_max_val = np.max(non_bg_vals, axis=0)
            # non_bg_max_idx = top2_idx[np.argmax(non_bg_vals, axis=0), np.arange(H * W)]

            # # result
            # result_flat = np.zeros(H * W, dtype=np.uint8)
            # # case 1: top2 includes bg and a strong enough non-bg class
            # mask1 = (is_bg_top2) & (non_bg_max_val > threshold)
            # result_flat[mask1] = non_bg_max_idx[mask1]
            # # case 2: top2 are both non-bg and one passes threshold
            # both_non_bg = (top2_idx != 0).all(axis=0)
            # top_val = top2_vals[np.argmax(top2_vals, axis=0), np.arange(H * W)]
            # top_idx = top2_idx[np.argmax(top2_vals, axis=0), np.arange(H * W)]
            # mask2 = (both_non_bg) & (top_val > threshold)
            # result_flat[mask2] = top_idx[mask2]

            # # reshape to (H, W)
            # result = result_flat.reshape(H, W)
            
            # pred_sem_seg = result
                            
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
            all_logits = np.stack(seg_logits, axis=0)  # (C, H, W)
            global_min = np.min(all_logits)
            global_max = np.max(all_logits)
            for logit_idx, (seg_logit, class_name) in enumerate(zip(seg_logits, ('background', ) + classes)):
                origin = 100, 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_img_height = 50
                txt_img = np.zeros((txt_img_height, input_width, 3), np.uint8)
                cv2.putText(txt_img, class_name, origin, font, 1, (255, 255, 255), 1)
                
                heatmap = (seg_logit - global_min) / (global_max - global_min + 1e-5)  # [0,1]
                heatmap = np.uint8(heatmap * 255)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                heatmap = cv2.vconcat([txt_img, heatmap])
                heatmaps.append(heatmap)

            colorbar = np.linspace(255, 0, input_height + txt_img_height, dtype=np.uint8).reshape(input_height + txt_img_height, 1)
            colorbar = cv2.applyColorMap(colorbar, cv2.COLORMAP_JET) 
            colorbar = cv2.resize(colorbar, (20, input_height + txt_img_height)) 
            gap = np.zeros((input_height + txt_img_height, 30, 3), dtype=np.uint8)  # 검정색 여백

            if gt_vis_img is not None:
                origin = 100, 25
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_img_height = 50
                txt_img = np.zeros((txt_img_height, input_width, 3), np.uint8)
                cv2.putText(txt_img, "GT", origin, font, 1, (255, 255, 255), 1)
                gt_vis_heatmap = cv2.vconcat([txt_img, gt_vis_img])
                vis_heatmap = cv2.hconcat([gt_vis_heatmap] + heatmaps + [gap, colorbar, gap])
            else:
                vis_heatmap = cv2.hconcat(heatmaps + [gap, colorbar, gap])
                
            if parent_path:
                if create_parent_path:
                    if not osp.exists(osp.join(logits_dir, parent_path)):
                        os.makedirs(osp.join(logits_dir, parent_path))
                    cv2.imwrite(osp.join(logits_dir, parent_path, filename + f'_{idx}_{jdx}.png'), vis_heatmap)   
                else:
                    cv2.imwrite(osp.join(logits_dir, '_'.join(parent_path.split('/')).replace('._', '') + '_' + filename + f'_{idx}_{jdx}.png'), vis_heatmap)
            else:
                cv2.imwrite(osp.join(logits_dir, filename + f'_{idx}_{jdx}.png'), vis_heatmap)
            
            