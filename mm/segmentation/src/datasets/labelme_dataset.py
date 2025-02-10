from typing import List
import os.path as osp

import json
from mmseg.registry import DATASETS
from mmseg.datasets.basesegdataset import BaseSegDataset
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
                 mode,
                 patch=None,
                 rois=[[]],
                 annotate=False,
                 logs_dir=None,
                 img_suffix='.bmp',
                 seg_map_suffix='.json',
                 reduce_zero_label=False,
                 **kwargs) -> None:
        self.METAINFO.update({'classes': tuple(classes), 'palette': self._palette[:len(tuple(classes))]})
        self.CLASSES = tuple(classes)
        self.PALETTE = self._palette[:len(tuple(classes))]
        self._mode = mode 
        
        assert self._mode in ['train', 'val', 'test'], ValueError(f'[ERROR] There is no such mode for dataset: {self._mode}')
        
        self._rois = rois
        self._patch = patch
        self._annotate = annotate
        self._logs_dir = logs_dir
        
        super().__init__(img_suffix=img_suffix,
                 seg_map_suffix=seg_map_suffix,
                 reduce_zero_label=reduce_zero_label,
                 **kwargs)
        
    @property 
    def mode(self):
        return self._mode
        
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

        input_dir = self.data_prefix.get('img_path', None)
        data_list = []
        _suffix_len = len(self.img_suffix)

        if not self.patch['use_patch']:
            for img in fileio.list_dir_or_file(
                    dir_path=input_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                data_info = dict(img_path=osp.join(input_dir, img))
                seg_map = img[:-_suffix_len] + self.seg_map_suffix
                data_info['seg_map_path'] = osp.join(input_dir, seg_map)
                for roi in self._rois:
                    data_info['label_map'] = self.label_map
                    data_info['reduce_zero_label'] = self.reduce_zero_label
                    data_info['seg_fields'] = []
                    data_info['classes'] = self.CLASSES
                    data_info['mode'] = self._mode
                    data_info['roi'] = roi
                    data_list.append(data_info)
            data_list = sorted(data_list, key=lambda x: x['img_path'])
            return data_list
        else:
            if self._mode in ['train', 'val']:
                from visionsuite.engines.data.slicer import Slicer
                
                slicer = Slicer(
                self._mode,
                osp.join(input_dir, '..'),
                classes=list(self.CLASSES),
                # img_exts=[suffix[1:] for suffix in self.img_suffix],
                img_exts=[self.img_suffix[1:]],
                roi_info=self._rois,
                roi_from_json=False,
                patch_info=self.patch,
                logger=None,
                )
                slicer.run()
                slicer.save_imgs_info(output_dir='/HDD/etc/etc')
                imgs_info, num_data = slicer.imgs_info, slicer.num_data
                assert (
                    num_data != 0
                ), f"There is NO images in dataset directory for [{self._mode}]: {osp.join(input_dir, self._mode)} with {self.img_suffix}"

                for img_info in imgs_info:
                    for roi in img_info['patches']:
                        
                        data_info = dict(img_path=img_info['img_file'])
                        assert osp.exists(img_info['img_file']), ValueError(f"[ERROR] There is no such image file: {img_info['img_file']}")
                        data_info['seg_map_path'] = img_info['img_file'][:-_suffix_len] + self.seg_map_suffix
                        assert osp.exists(data_info['seg_map_path']), ValueError(f"[ERROR] There is no such json file: {data_info['seg_map_path']}")
                        
                        data_info['label_map'] = self.label_map
                        data_info['reduce_zero_label'] = self.reduce_zero_label
                        data_info['seg_fields'] = []
                        data_info['classes'] = self.CLASSES
                        data_info['mode'] = self._mode
                        data_info['roi'] = roi
                        data_list.append(data_info)
                data_list = sorted(data_list, key=lambda x: x['img_path'])
                return data_list

            else:
                for img in fileio.list_dir_or_file(
                    dir_path=input_dir,
                    list_dir=False,
                    suffix=self.img_suffix,
                    recursive=True,
                    backend_args=self.backend_args):
                    
                    dx = int((1.0 - self._patch['overlap_ratio']) * self._patch['width'])
                    dy = int((1.0 - self._patch['overlap_ratio']) * self._patch['height'])
                    
                    for roi in self._rois:
                        if len(roi) == 0:
                            
                            if osp.exists(osp.join(input_dir, img[:-_suffix_len] + self.seg_map_suffix)):
                                with open(osp.join(input_dir, img[:-_suffix_len] + self.seg_map_suffix), 'r') as jf:
                                    anns = json.load(jf)
                                    
                                width, height = anns['imageWidth'], anns['imageHeight']
                                do_metric = True
                            else:
                                from mm.utils.fileio.parse_image_file import get_image_size
                                width, height = get_image_size(osp.join(input_dir, img))
                                do_metric = False
                                
                            roi = [0, 0, width, height]

                        for y0 in range(roi[1], roi[3], dy):
                            for x0 in range(roi[0], roi[2], dx):
                                
                                if y0 + self._patch['height'] > roi[3] - roi[1]:
                                    y = roi[3] - roi[1] - self._patch['height']
                                else:
                                    y = y0

                                if x0 + self._patch['width'] > roi[2] - roi[0]:
                                    x = roi[2] - roi[0] - self._patch['width']
                                else:
                                    x = x0
                                    
                                data_info = dict(img_path=osp.join(input_dir, img))
                                data_info['seg_map_path'] = osp.join(input_dir, img[:-_suffix_len] + self.seg_map_suffix)
                                data_info['label_map'] = self.label_map
                                data_info['reduce_zero_label'] = self.reduce_zero_label
                                data_info['seg_fields'] = []
                                data_info['classes'] = self.CLASSES
                                data_info['mode'] = self._mode
                                data_info['roi'] = [x, y, x + self._patch['width'], y + self._patch['height']]
                                data_info['do_metric'] = do_metric
                                data_info['annotate'] = self._annotate
                                data_list.append(data_info)
                                
                data_list = sorted(data_list, key=lambda x: x['img_path'])
                return data_list
                