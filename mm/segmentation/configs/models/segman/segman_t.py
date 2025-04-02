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
        type='SegMANEncoder_t',
        pretrained=None,
        style='pytorch',
        num_classes=None,
        frozen_stages=-1,
    ),
    decode_head=dict(
        type='SegMANDecoder',
        in_channels=[32, 64, 144, 192],
        in_index=[0, 1, 2, 3],
        channels=128,
        feat_proj_dim=192,
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

# learning policy
param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=1e-6,
        by_epoch=False,
        begin=0,
        end=1500  # warmup_iters
    ),
    dict(
        type='PolyLR',
        power=1.0,
        eta_min=1e-8,
        by_epoch=False
    )
]

evaluation = dict(interval=4000, metric='mIoU',save_best='mIoU')