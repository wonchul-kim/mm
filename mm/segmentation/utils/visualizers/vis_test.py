import os
import os.path as osp
import imgviz
import numpy as np
import cv2

from visionsuite.utils.dataset.formats.labelme.utils import get_labeled_gt_image, get_points_from_image, get_roi_info, init_labelme_json

def vis_test(outputs, output_dir, idx):
    
    color_map = imgviz.label_colormap(50)
    if not osp.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
        
        
    for jdx, output in enumerate(outputs):
        # img
        img_path = output.img_path 
        filename = osp.split(osp.splitext(img_path)[0])[-1]
        
        output_dir = osp.join(output_dir, filename)
        if not osp.exists(output_dir):
            os.mkdir(output_dir)         
        
        img = cv2.imread(img_path)
        img_shape = img.shape 
        
        # outputs
        seg_logits = output.seg_logits.data.cpu().detach().numpy()
        gt_sem_seg = output.gt_sem_seg.data.squeeze(0).cpu().detach().numpy()
        pred_sem_seg = output.pred_sem_seg.data.squeeze(0).cpu().detach().numpy()
        reduce_zero_label = output.reduce_zero_label

        # if output.annotate and not osp.exists(output.seg_map_path):
        #     _labelme = init_labelme_json(filename + ".bmp", img_shape[1], img_shape[0])
        #     _labelme = get_points_from_image(pred_sem_seg, output.classes,
        #                         _roi_info,
        #                         [x, y],
        #                         _labelme,
        #                         self.configs.test.contour_thres,
        #                     )
        # else:
        #     _labelme = None

        # roi
        if len(output.roi) == 0:
            roi = [0, 0, img_shape[1], img_shape[0]]
        else:
            roi = output.roi
            
        if reduce_zero_label:
            gt_sem_seg[gt_sem_seg == 255] = -1
            gt_sem_seg += 1
    
        if output.do_metric:
            gt_vis_img = img.copy()
            gt_vis_img[roi[1]:roi[3], roi[0]:roi[2]] = cv2.addWeighted(img[roi[1]:roi[3], roi[0]:roi[2]], 0.4, color_map[gt_sem_seg], 0.6, 0)
            cv2.rectangle(gt_vis_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)
        else:
            gt_vis_img = None 
            
        pred_vis_img = img.copy()
        pred_vis_img[roi[1]:roi[3], roi[0]:roi[2]] = cv2.addWeighted(img[roi[1]:roi[3], roi[0]:roi[2]], 0.4, color_map[pred_sem_seg], 0.6, 0)
        cv2.rectangle(pred_vis_img, (roi[0], roi[1]), (roi[2], roi[3]), (0, 0, 255), 2)

        if gt_vis_img is not None:
            vis_img = np.hstack((gt_vis_img, pred_vis_img))
        else:
            vis_img = pred_vis_img
        cv2.imwrite(osp.join(output_dir, filename + f'_{idx}_{jdx}.png'), vis_img)
        
            
        