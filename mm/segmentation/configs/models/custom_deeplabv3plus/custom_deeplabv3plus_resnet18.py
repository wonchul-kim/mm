_base_ = [
        '../../_base_/default_runtime.py', 
        '../../_base_/datasets/labelme.py'
]

crop_size = (512, 512)
num_classes = 3
max_iters = 120000
norm_cfg = dict(type='BN', requires_grad=True)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='CustomDeeplabv3plus',
        num_classes=num_classes,
    ),
    decode_head=dict(
        type='EmptyDecodeHead',
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.5),
            dict(
                type='SoftmaxFocalLoss',
                gamma=2.0,
                alpha=0.25,
                loss_weight=0.5,
                class_weight=[1.0]*num_classes,
                ),
        ]
    ),
    train_cfg=dict(),
    test_cfg=dict(mode='whole')
)

optimizer = dict(type='AdamW', lr=0.001, weight_decay=5e-4)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
)
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=1e-7,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False)
]