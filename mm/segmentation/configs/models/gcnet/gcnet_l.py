_base_ = [
        './gcnet_s.py'
]
crop_size = (1024, 1024)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    size=crop_size,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255
)
num_classes = 19
max_iters = 120000
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='GCNet',
        in_channels=3,
        channels=64,
        num_blocks_per_stage=[5, 5, [5, 5], [5, 5], [3, 3]],
    ),
    decode_head=dict(
        type='GCNetHead',
        in_channels=64 * 4,
        channels=256,
        dropout_ratio=0.,
        num_classes=num_classes,
        loss_decode=[
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=[1.]*num_classes,
                loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=[1.]*num_classes,
                loss_weight=1.0),
        ]
    ),
)

optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0005)
optim_wrapper = dict(type='OptimWrapper', optimizer=optimizer, clip_grad=None)
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0.,
        power=0.9,
        begin=0,
        end=max_iters,
        by_epoch=False)
]