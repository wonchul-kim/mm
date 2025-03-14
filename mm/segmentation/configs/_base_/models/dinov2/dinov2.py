crop_size = (518, 518)
stride = (crop_size[0] // 2, crop_size[1] // 2)
data_preprocessor = dict(
    bgr_to_rgb=True,
    mean=[123.675, 116.28, 103.53,],
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    std=[58.395, 57.12, 57.375,],
    type='SegDataPreProcessor')

model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='DinoVisionBackbone',
        size='base',
        img_size=crop_size,
        patch_size=14,
        freeze_vit=True,
        init_cfg=dict(type='Pretrained', checkpoint="/HDD/weights/dinov2/dinov2_vitb14_pretrain.pth"),
        # norm_cfg=norm_cfg,
        # out_indices=[8, 9, 10, 11]
        ),
    train_cfg=dict(),
    test_cfg=dict(mode='slide', crop_size=crop_size, stride=stride),
)
