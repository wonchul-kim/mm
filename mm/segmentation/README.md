# MMSegmentation 

## Envs.

#### pytorch
```shell
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

#### mim list
```shell
mmcv            2.1.0      https://github.com/open-mmlab/mmcv
mmdet           3.3.0      https://github.com/open-mmlab/mmdetection
mmengine        0.10.6     https://github.com/open-mmlab/mmengine
mmsegmentation  1.2.2      https://github.com/open-mmlab/mmsegmentation
```

## Models

### ADE20K

#### SegMAN

| Method        | Backbone         | Crop Size   | Params (M)| Mem (GB)  | Inf time (fps)   | Device   | mIoU    | mIoU(ms+flip)  | config  |
| :-----------: | :--------------: | :---------: | :-------: | :-------: | :--------------: | :------: | :-----: | :------------: | :-----: | 
| SegMAN        | T                | 512x512     | 6.4       |           |                  |          | 43.0    |              - |         | 
| SegMAN        | S                | 512x512     | 29.4      |           |                  |          | 51.3    |              - |         | 
| SegMAN        | B                | 512x512     | 51.8      |           |                  |          | 52.6    |              - |         | 

#### Mask2Former

| Method        | Backbone         | Crop Size   | Lr schd   | Mem (GB)  | Inf time (fps)   | Device   | mIoU    | mIoU(ms+flip)  | config  |
| :-----------: | :--------------: | :---------: | :-------: | :-------: | :--------------: | :------: | :-----: | :------------: | :-----: | 
| Mask2Former   | R-50-D32         | 512x512     | 160000    |     3.31  | 26.59            | A100     | 47.87   |              - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_r50_8xb2-160k_ade20k-512x512.py) | 
| Mask2Former   | R-101-D32        | 512x512     | 160000    |     4.09  | 22.97            | A100     | 48.60   |              - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_r101_8xb2-160k_ade20k-512x512.py) | 
| Mask2Former   | Swin-T           | 512x512     | 160000    |     3826  | 23.82            | A100     | 48.66   |              - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-t_8xb2-160k_ade20k-512x512.py) | 
| Mask2Former   | Swin-S           | 512x512     | 160000    |     3.74  | 19.69            | A100     | 51.24   |              - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-s_8xb2-160k_ade20k-512x512.py) | 
| Mask2Former   | Swin-B           | 640x640     | 160000    |     5.66  | 12.48            | A100     | 52.44   |              - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-b-in1k-384x384-pre_8xb2-160k_ade20k-640x640.py) | 
| Mask2Former   | Swin-B (in22k)   | 640x640     | 160000    |     5.66  | 12.43            | A100     | 53.90   |              - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py) | 
| Mask2Former   | Swin-L (in22k)   | 640x640     | 160000    |     8.86  | 8.81             | A100     | 56.01   |              - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py) | 

#### Deeplabv3+

| Method       | Backbone   | Crop Size   | Lr schd  | Mem (GB)   | Inf time (fps)   | Device   |  mIoU  | mIoU(ms+flip)  | config  |
| :----------: | :--------: | :---------: | :------: | :--------: | :--------------: | :------: | :----: | :------------: | :-----: |
| DeepLabV3+   | R-50-D8    | 512x512     |   80000  | 10.6       | 21.01            | V100     | 42.72  |         43.75  | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-80k_ade20k-512x512.py)   | 
| DeepLabV3+   | R-101-D8   | 512x512     |   80000  | 14.1       | 14.16            | V100     | 44.60  |         46.06  | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py) | 
| DeepLabV3+   | R-50-D8    | 512x512     |  160000  | -          | -                | V100     | 43.95  |         44.93  | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/deeplabv3plus/deeplabv3plus_r50-d8_4xb4-160k_ade20k-512x512.py)  | 
| DeepLabV3+   | R-101-D8   | 512x512     |  160000  | -          | -                | V100     | 45.47  |         46.35  | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/deeplabv3plus/deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py) | 


------------------------------------------------------------------------------------------------
### Cityscapes 

#### SegMAN

| Method        | Backbone         | Crop Size   | Params (M)| Mem (GB)  | Inf time (fps)   | Device   | mIoU    | mIoU(ms+flip)  | config  |
| :-----------: | :--------------: | :---------: | :-------: | :-------: | :--------------: | :------: | :-----: | :------------: | :-----: | 
| SegMAN        | T                | 512x512     | 6.4       |           |                  |          | 80.3    |              - |         | 
| SegMAN        | S                | 512x512     | 29.4      |           |                  |          | 83.2    |              - |         | 
| SegMAN        | B                | 512x512     | 51.8      |           |                  |          | 83.8    |              - |         | 

#### Mask2Former
| Method       | Backbone         | Crop Size   | Lr schd   | Mem (GB)  | Inf time (fps)   | Device   | mIoU    | mIoU(ms+flip)  | config |
| :----------: | :--------------: | :---------: | :-------: | :-------: | :--------------: | :------: | :-----: | :------------: | :----: |
| Mask2Former | R-50-D32       | 512x1024  | 90000   |     5.67 | 9.17           | A100   | 80.44 |             - |                      [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | R-101-D32      | 512x1024  | 90000   |     6.81 | 7.11           | A100   | 80.80 |             - |                     [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | Swin-T         | 512x1024  | 90000   |     6.36 | 7.18           | A100   | 81.71 |             - |                   [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | Swin-S         | 512x1024  | 90000   |     8.09 | 5.57           | A100   | 82.57 |             - |                   [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | Swin-B (in22k) | 512x1024  | 90000   |    10.89 | 4.32           | A100   | 83.52 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | Swin-L (in22k) | 512x1024  | 90000   |    15.83 | 2.86           | A100   | 83.65 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py) | 


### COCO-stuff 

#### SegMAN

| Method        | Backbone         | Crop Size   | Params (M)| Mem (GB)  | Inf time (fps)   | Device   | mIoU    | mIoU(ms+flip)  | config  |
| :-----------: | :--------------: | :---------: | :-------: | :-------: | :--------------: | :------: | :-----: | :------------: | :-----: | 
| SegMAN        | T                | 512x512     | 6.4       |           |                  |          | 41.3    |              - |         | 
| SegMAN        | S                | 512x512     | 29.4      |           |                  |          | 47.5    |              - |         | 
| SegMAN        | B                | 512x512     | 51.8      |           |                  |          | 48.3    |              - |         | 


### Build docker image from `Dockerfile`
```sh
docker build -t mmsegmentation docker/
```

### Pull docke rimage
```sh
docker pull onedang2/mmengine-0.10.5-mmcv2.0.1-mmdet3.3.0-mmseg1.2.2
```



### Models

#### SegMAN

* Required libraries to install
    - [VMamba](https://github.com/MzeroMiko/VMamba)
        ```shell
        git clone git@github.com:MzeroMiko/VMamba.git
        cd cd VMamba/kernels/selective_scan
        pip install .
        ```
    - [Natten](https://github.com/SHI-Labs/NATTEN)
        ```shell
        pip install natten==0.17.3+torch210cu118 -f https://shi-labs.com/natten/wheels/cu118/torch2.1.0/natten-0.17.3+torch210cu118-cp310-cp310-linux_x86_64.whl
        ```

        > If you have any other version of torch and cuda, NEED to check and synch them by checking https://shi-labs.com/natten/wheels/.

#### [Efficient Mirror Detection via Multi-level Heterogeneous Learning (HetNet) (AAAI23, ORAL)](https://arxiv.org/pdf/2211.15644v1)



## TODO:
| 모델명        | 연도 | 코드 공개 | Citation/DOI | 특징 및 비고 |
|--------------|------|-----------|--------------|--------------|
| **SegFormer** | 2021 | O | [arXiv:2105.15203](https://arxiv.org/abs/2105.15203)<br>[GitHub](https://github.com/NVlabs/SegFormer) | 경량/고성능 트랜스포머 세그멘테이션, MLP 디코더, 실시간/엣지 디바이스 적합 |
| **Mask2Former** | 2021 | O | [CVPR 2022](https://arxiv.org/abs/2112.01527)<br>[GitHub](https://github.com/facebookresearch/Mask2Former) | 범용 세그멘테이션(SOTA), Masked Attention, 높은 정확도, 무겁고 연산량 많음 |
| **PEM** | 2024 | O | [CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Cavagnero_PEM_Prototype-based_Efficient_MaskFormer_for_Image_Segmentation_CVPR_2024_paper.pdf) | Mask2Former 구조 경량화, 프로토타입 기반, SOTA에 근접한 성능 |
| **AFFormer** | 2023 | O | [arXiv:2301.04648](https://arxiv.org/abs/2301.04648) | 디코더 제거, 병렬 구조, 3M 파라미터로 ADE20K 등에서 경량/고성능 |
| **PlainSeg/PlainSeg-Hier** | 2023 | O | [arXiv:2310.12755](https://arxiv.org/abs/2310.12755) | Plain ViT+3x3 conv만 추가, 단순/경량 구조, 다양한 데이터셋에서 준수한 성능 |
| **ExtremeMETA** | 2025 | 미확인 | [J. Imaging Sci. Technol. 2025, 69(4): 040403](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/jist/0/0/040403) | 대형 커널, 모델 압축, 95% mIoU 달성, 실시간/고효율 |
