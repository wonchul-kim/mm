_base_ = [
    '../../_base_/models/segman/segman.py',
    # '../../_base_/datasets/ade20k_repeat.py',
    '../../_base_/datasets/labelme.py',
    '../../_base_/default_runtime.py',
    '../../_base_/schedules/schedule_160k_adamw.py'
]

# model settings
norm_cfg = dict(type='SyncBN', requires_grad=True)
crop_size = (512, 512)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255,
    size=crop_size,
    test_cfg=dict(size_divisor=32))
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
    type='SegMANEncoder_s',
    pretrained='/HDD/weights/segman/SegMAN_Encoder_s.pth.tar',
    style='pytorch',
    num_classes=None,
    ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[64, 144, 288, 512],
        in_index=[0, 1, 2, 3],
        channels=144,
        feat_proj_dim=288,
        dropout_ratio=0.1,
        num_classes=None,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg = dict(mode='whole'))

# optimizer
optimizer = dict(type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, 
                     clip_grad=None,
                     paramwise_cfg=dict(custom_keys={'pos_block': dict(decay_mult=0.),
                                                 'norm': dict(decay_mult=0.),
                                                 'head': dict(lr_mult=10.)
                                                 }))
lr_config = dict(policy='poly',
                 warmup='linear',
                 warmup_iters=1500,
                 warmup_ratio=1e-6,
                 power=1.0, min_lr=0.0, by_epoch=False)


evaluation = dict(interval=8000, metric='mIoU')