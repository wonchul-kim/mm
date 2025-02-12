_base_ = ['./mask2former_swin-b-in1k.py']

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_base_patch4_window12_384_22k_20220317-e5c09f74.pth'  # noqa
model = dict(
    backbone=dict(init_cfg=dict(type='Pretrained', checkpoint=pretrained)))