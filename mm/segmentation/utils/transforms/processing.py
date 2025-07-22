from mmcv.transforms.processing import Resize
from mmcv.transforms.builder import TRANSFORMS

@TRANSFORMS.register_module()
class ResizeWithPatch(Resize):
    def transform(self, results: dict) -> dict:
        """Transform function to resize images, bounding boxes, semantic
        segmentation map and keypoints.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Resized results, 'img', 'gt_bboxes', 'gt_seg_map',
            'gt_keypoints', 'scale', 'scale_factor', 'img_shape',
            and 'keep_ratio' keys are updated in result dict.
        """
        if 'patch' in results and results['patch']['use_patch']:
            if self.scale:
                self.scale = (results['img_shape'][1], results['img_shape'][0])
        if self.scale:
            results['scale'] = self.scale
        else:
            img_shape = results['img'].shape[:2]
            results['scale'] = self._scale_size(img_shape[::-1],
                                        self.scale_factor)  # type: ignore
        self._resize_img(results)
        self._resize_bboxes(results)
        self._resize_seg(results)
        self._resize_keypoints(results)
        return results
