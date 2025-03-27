from mmdet.datasets.builder import PIPELINES
import numpy as np
import copy

@PIPELINES.register_module()
class MultiROICrop:
    def __init__(self, rois):
        """
        rois: [(x_min, y_min, x_max, y_max), ...] 형태의 리스트
        """
        self.rois = rois

    def __call__(self, results):
        img = results['img']
        h, w, _ = img.shape
        cropped_results = []

        for x_min, y_min, x_max, y_max in self.rois:
            x_min, y_min = max(0, x_min), max(0, y_min)
            x_max, y_max = min(w, x_max), min(h, y_max)

            # 개별 샘플을 생성
            new_results = copy.deepcopy(results)
            new_results['img'] = img[y_min:y_max, x_min:x_max]
            new_results['img_shape'] = new_results['img'].shape[:2]  # 업데이트된 이미지 크기 반영

            # 바운딩 박스도 ROI 내부에 맞게 조정
            if 'gt_bboxes' in new_results and len(new_results['gt_bboxes']) > 0:
                new_bboxes = []
                new_labels = []
                for bbox, label in zip(new_results['gt_bboxes'], new_results['gt_labels']):
                    x1, y1, x2, y2 = bbox
                    if x1 >= x_max or x2 <= x_min or y1 >= y_max or y2 <= y_min:
                        continue  # ROI 영역 밖의 바운딩 박스는 제거
                    
                    # ROI 좌표 기준으로 변환
                    new_x1 = max(0, x1 - x_min)
                    new_y1 = max(0, y1 - y_min)
                    new_x2 = min(x_max - x_min, x2 - x_min)
                    new_y2 = min(y_max - y_min, y2 - y_min)
                    new_bboxes.append([new_x1, new_y1, new_x2, new_y2])
                    new_labels.append(label)

                new_results['gt_bboxes'] = np.array(new_bboxes, dtype=np.float32)
                new_results['gt_labels'] = np.array(new_labels, dtype=np.int64)

            cropped_results.append(new_results)

        return cropped_results  # 여러 개의 샘플을 반환하여 개별 학습
