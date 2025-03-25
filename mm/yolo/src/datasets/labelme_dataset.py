# Copyright (c) OpenMMLab. All rights reserved.
from glob import glob 
import os.path as osp
from typing import List, Union
import json 

from mmdet.registry import DATASETS
from mmdet.datasets.base_det_dataset import BaseDetDataset


@DATASETS.register_module()
class LabelmeDataset(BaseDetDataset):
    METAINFO = dict()
    _palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]]

    def __init__(self,
                 classes, 
                 mode,
                 patch=None,
                 rois=[[]],
                 annotate=False,
                 logs_dir=None,
                 img_suffix='.bmp',
                 ann_suffix='.json',
                 **kwargs) -> None:
        self.METAINFO.update({'classes': tuple(classes), 'palette': self._palette[:len(tuple(classes))]})
        self.CLASSES = tuple(classes)
        self.PALETTE = self._palette[:len(tuple(classes))]
        self.class2label = {val: key for key, val in enumerate(self.CLASSES)}
        self._mode = mode 
        self._img_suffix = img_suffix 
        self._ann_suffix = ann_suffix

        
        assert self._mode in ['train', 'val', 'test'], ValueError(f'[ERROR] There is no such mode for dataset: {self._mode}')
        
        self._rois = rois
        self._patch = patch
        self._annotate = annotate
        self._logs_dir = logs_dir
        
        super().__init__(**kwargs)
        
    @property 
    def mode(self):
        return self._mode
    
    @property 
    def img_suffix(self):
        return self._img_suffix
    
    @property 
    def ann_suffix(self):
        return self._ann_suffix
        
    @property 
    def rois(self):
        return self._rois
    
    @property 
    def patch(self):
        return self._patch
        
    @property 
    def annotate(self):
        return self._annotate
    
    @property 
    def logs_dir(self):
        return self._logs_dir
    
    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  
        self.img_files = glob(osp.join(self.data_prefix['img'], f'*{self.img_suffix}'))
        data_list = []
        total_ann_ids = []
        img_info = {}
        for img_file in self.img_files:
            ann_file = osp.splitext(img_file)[0] + self.ann_suffix
            assert osp.exists(ann_file), RuntimeError(f"There is no such annotation file: {ann_file}")
            
            with open(ann_file, 'r') as jf:
                ann_info = json.load(jf)
            
            img_info['img_path'] = img_file 
            img_info['height'] = ann_info['imageHeight']
            img_info['width'] = ann_info['imageWidth']
            
            parsed_data_info = self.parse_data_info({
                'raw_ann_info': ann_info,
                'raw_img_info': img_info
            })
            data_list.append(parsed_data_info)

        return data_list

    def parse_data_info(self, raw_data_info: dict) -> Union[dict, List[dict]]:
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']

        data_info = {}

        # TODO: need to change data_prefix['img'] to data_prefix['img_path']
        seg_map_path = None
        data_info['img_path'] = img_info['img_path']
        # data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        instances = []
        for i, ann in enumerate(ann_info['shapes']):
            instance = {}

            label = ann['label']
            if label in self.CLASSES or label.lower() in self.CLASSES:
                shape_type = ann['shape_type']
                points = ann['points']
                
                if shape_type in ['polygon', 'watershed', 'point', 'rotatedrect', 'rectangle', 'circle']:
                    
                    if shape_type in ['polygon', 'watershed', 'point'] and len(points) <= 2:
                        continue 
                    else:
                        xs, ys = [], []
                        for point in points:
                            xs.append(point[0])
                            ys.append(point[1])
                        
                        x1, y1, w, h = min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)
                else:
                    raise NotImplementedError(f'NOT yet Considered shape-type: {shape_type}')
                    
                instance['bbox'] = [x1, y1, x1 + w, y1 + h]
                instance['bbox_label'] = int(self.class2label[label])
                instance['ignore_flag'] = 0
            
                instances.append(instance)
                
        data_info['instances'] = instances
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter annotations according to filter_cfg.

        Returns:
            List[dict]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        if self.filter_cfg is None:
            return self.data_list

        filter_empty_gt = self.filter_cfg.get('filter_empty_gt', False)
        min_size = self.filter_cfg.get('min_size', 0)

        # obtain images that contain annotation
        ids_with_ann = set(data_info['img_id'] for data_info in self.data_list)
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_data_infos = []
        for i, data_info in enumerate(self.data_list):
            img_id = data_info['img_id']
            width = data_info['width']
            height = data_info['height']
            if filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(width, height) >= min_size:
                valid_data_infos.append(data_info)

        return valid_data_infos