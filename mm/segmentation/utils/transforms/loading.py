# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import mmengine.fileio as fileio
import numpy as np

import mmcv
from mmcv.transforms.builder import TRANSFORMS
from mmcv.transforms.loading import LoadImageFromFile
from mmseg.datasets.transforms import LoadAnnotations
from mm.utils.fileio.parse_image_file import get_image_size


@TRANSFORMS.register_module()
class LoadImageFromFileWithRoi(LoadImageFromFile):

    def transform(self, results: dict) -> Optional[dict]:
        filename = results['img_path']
        try:
            if self.file_client_args is not None:
                file_client = fileio.FileClient.infer_client(
                    self.file_client_args, filename)
                img_bytes = file_client.get(filename)
            else:
                img_bytes = fileio.get(
                    filename, backend_args=self.backend_args)
            img = mmcv.imfrombytes(
                img_bytes, flag=self.color_type, backend=self.imdecode_backend)
        except Exception as e:
            if self.ignore_empty:
                return None
            else:
                raise e
        # in some cases, images are not read successfully, the img would be
        # `None`, refer to https://github.com/open-mmlab/mmpretrain/issues/1427
        assert img is not None, f'failed to load image: {filename}'
        if self.to_float32:
            img = img.astype(np.float32)

        if 'roi' in results and results['roi'] != []:
            img = img[results['roi'][1]:results['roi'][3], results['roi'][0]:results['roi'][2], :]

        results['img'] = img
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'ignore_empty={self.ignore_empty}, '
                    f'to_float32={self.to_float32}, '
                    f"color_type='{self.color_type}', "
                    f"imdecode_backend='{self.imdecode_backend}', ")

        if self.file_client_args is not None:
            repr_str += f'file_client_args={self.file_client_args})'
        else:
            repr_str += f'backend_args={self.backend_args})'

        return repr_str


@TRANSFORMS.register_module()
class LoadLabelmeAnnotations(LoadAnnotations):
    def _load_seg_map(self, results: dict) -> None:

        width, height = get_image_size(osp.join(osp.dirname(osp.abspath(results['seg_map_path'])), results['img_path']))
        if osp.exists(results['seg_map_path']):
            gt_semantic_seg = get_mask_from_labelme(results['mode'], results['seg_map_path'], 
                                                width=width, height=height,
                                                format='opencv',
                                    class2label={key.lower(): val for val, key in enumerate(results['classes'])}).astype(np.uint8)
        else:
            gt_semantic_seg = np.zeros((height, width))

        if 'roi' in results and results['roi'] != []:
            gt_semantic_seg = gt_semantic_seg[results['roi'][1]:results['roi'][3], results['roi'][0]:results['roi'][2]]

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


import numpy as np
import json
import cv2 
import os.path as osp
import math

def get_mask_from_labelme(mode, json_file, class2label, width=None, height=None, format='pil', metis=None):
    if mode in ['train', 'val']:
        
        assert osp.exists(json_file), ValueError(f"There is no annotation file: {json_file} to {mode}")
        if metis is None:
            with open(json_file, encoding='utf-8') as f:
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
    elif mode == 'test':
        if osp.exists(json_file):
            if metis is None:
                with open(json_file, encoding='utf-8') as f:
                    anns = json.load(f)
            else:
                anns = {"shapes": metis}
                
                
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
        else:
            mask = np.zeros((height, width))

    if format == 'pil':
        from PIL import Image
        
        return Image.fromarray(mask)
    elif format == 'opencv':
        return mask
    else:
        NotImplementedError(f'There is no such case for {format}')
