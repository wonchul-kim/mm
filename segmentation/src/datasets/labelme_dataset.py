from mmseg.registry import DATASETS, TRANSFORMS
from mmseg.datasets.transforms import LoadAnnotations
from mmseg.datasets.basesegdataset import BaseSegDataset
from typing import List
import os.path as osp
import mmengine.fileio as fileio


@DATASETS.register_module()
class LabelmeDataset(BaseSegDataset):
    """
    In segmentation map annotation for MaskDataset, 0 stands for background, which
    is not included in defined categories. ``reduce_zero_label`` is fixed to True.
    The ``img_suffix`` is fixed to '.jpg' and ``seg_map_suffix`` is fixed to
    '.png'.
    """
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
                 img_suffix='.bmp',
                 seg_map_suffix='.bmp',
                 reduce_zero_label=True,
                 **kwargs) -> None:
        
        self.METAINFO.update({'classes': tuple(classes), 'palette': self._palette[:len(tuple(classes))]})
        self.CLASSES = tuple(classes)
        self.PALETTE = self._palette[:len(tuple(classes))]
        
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory or annotation file.

        Returns:
            list[dict]: All data info of dataset.
        """
        data_list = []
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('seg_map_path', None)
        if not osp.isdir(self.ann_file) and self.ann_file:
            assert osp.isfile(self.ann_file), \
                f'Failed to load `ann_file` {self.ann_file}'
            lines = mmengine.list_from_file(
                self.ann_file, backend_args=self.backend_args)
            for line in lines:
                img_name = line.strip()
                data_info = dict(
                    img_path=osp.join(img_dir, img_name + self.img_suffix))
                if ann_dir is not None:
                    seg_map = img_name + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_list.append(data_info)
        else:
            _suffix_len = len(self.img_suffix)
            for img in fileio.list_dir_or_file(
                    dir_path=img_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(img_dir, img))
                if ann_dir is not None:
                    seg_map = img[:-_suffix_len] + self.seg_map_suffix
                    data_info['seg_map_path'] = osp.join(ann_dir, seg_map)
                data_info['label_map'] = self.label_map
                data_info['reduce_zero_label'] = self.reduce_zero_label
                data_info['seg_fields'] = []
                data_info['classes'] = self.CLASSES
                data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
        return data_list

import numpy as np
import json
import cv2 

def get_mask_from_labelme(json_file, class2label, width=None, height=None, format='pil', metis=None):

    if metis is None:
        with open(json_file) as f:
            anns = json.load(f)
    else:
        anns = {"shapes": metis}
        
    if height is None:
        height = anns['imageHeight']
    if width is None:
        width = anns['imageWidth']
    mask = np.zeros((height, width))
    for label_idx in range(0, len(class2label.keys())):
        for shapes in anns['shapes']:
            shape_type = shapes['shape_type'].lower()
            label = shapes['label'].lower()
            if label == list(class2label.keys())[label_idx]:
                _points = shapes['points']
                if shape_type == 'circle':
                    cx, cy = _points[0][0], _points[0][1]
                    radius = int(math.sqrt((cx - _points[1][0]) ** 2 + (cy - _points[1][1]) ** 2))
                    cv2.circle(mask, (int(cx), int(cy)), int(radius), True, -1)
                elif shape_type in ['rectangle']:
                    if len(_points) == 2:
                        arr = np.array(_points, dtype=np.int32)
                    else:
                        RuntimeError(f"Rectangle labeling should have 2 points")
                    cv2.fillPoly(mask, [arr], color=(class2label[label]))
                elif shape_type in ['polygon', 'watershed']:
                    if len(_points) > 2:  # handle cases for 1 point or 2 points
                        arr = np.array(_points, dtype=np.int32)
                    else:
                        continue
                    cv2.fillPoly(mask, [arr], color=(class2label[label]))
                elif shape_type in ['point']:
                    pass
                else:
                    raise ValueError(f"There is no such shape-type: {shape_type}")

    if format == 'pil':
        from PIL import Image
        
        return Image.fromarray(mask)
    elif format == 'opencv':
        return mask
    else:
        NotImplementedError(f'There is no such case for {format}')

@TRANSFORMS.register_module()
class LoadLabelmeAnnotations(LoadAnnotations):
    def _load_seg_map(self, results: dict) -> None:
        """Private function to load semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict contains loaded semantic segmentation annotations.
        """

        gt_semantic_seg = get_mask_from_labelme(results['seg_map_path'], width=1024, height=1024, format='opencv',
                                    class2label={key: val + 1 for val, key in enumerate(results['classes'])}).astype(np.uint8)

        if self.reduce_zero_label is None:
            self.reduce_zero_label = results['reduce_zero_label']
        assert self.reduce_zero_label == results['reduce_zero_label'], \
            'Initialize dataset with `reduce_zero_label` as ' \
            f'{results["reduce_zero_label"]} but when load annotation ' \
            f'the `reduce_zero_label` is {self.reduce_zero_label}'
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = 255
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == 254] = 255
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['seg_fields'].append('gt_seg_map')


