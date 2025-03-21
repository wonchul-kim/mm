# MMSegmentation 

## Envs.

#### pytorch
```
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
```

## Models

### ADE20K
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

#### Mask2Former
| Method       | Backbone         | Crop Size   | Lr schd   | Mem (GB)  | Inf time (fps)   | Device   | mIoU    | mIoU(ms+flip)  | config |
| :----------: | :--------------: | :---------: | :-------: | :-------: | :--------------: | :------: | :-----: | :------------: | :----: |
| Mask2Former | R-50-D32       | 512x1024  | 90000   |     5.67 | 9.17           | A100   | 80.44 |             - |                      [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_r50_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | R-101-D32      | 512x1024  | 90000   |     6.81 | 7.11           | A100   | 80.80 |             - |                     [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_r101_8xb2-90k_cityscapes-512x1024.py) | 
Former | Swin-T         | 512x1024  | 90000   |     6.36 | 7.18           | A100   | 81.71 |             - |                   [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-t_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | Swin-S         | 512x1024  | 90000   |     8.09 | 5.57           | A100   | 82.57 |             - |                   [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-s_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | Swin-B (in22k) | 512x1024  | 90000   |    10.89 | 4.32           | A100   | 83.52 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-b-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py) | 
| Mask2Former | Swin-L (in22k) | 512x1024  | 90000   |    15.83 | 2.86           | A100   | 83.65 |             - | [config](https://github.com/open-mmlab/mmsegmentation/blob/main/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-90k_cityscapes-512x1024.py) | 


### Build docker image from `Dockerfile`
```sh
docker build -t mmsegmentation docker/
```

### Pull docke rimage
```sh
docker pull onedang2/mmengine-0.10.5-mmcv2.0.1-mmdet3.3.0-mmseg1.2.2
```

