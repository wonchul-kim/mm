_base_ = [
    '../../_base_/models/cosnet/upernet_r50.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/labelme.py'
    # '/tmp/custom_dataset.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    pretrained=None,
    backbone=dict(
        type='COSNet',
        depths=[3, 3, 12, 3],
        style='pytorch'),
    decode_head=dict(num_classes=3,
                     in_channels=[72, 72*2, 72*4, 72*8],
                     channels=256,
                     in_index=[0, 1, 2, 3],
                     norm_cfg=norm_cfg),
    auxiliary_head=dict(num_classes=3,
                        in_channels=72*4,
                        in_index=2,
                        norm_cfg=norm_cfg)
    )

# optimizer
optimizer = dict(type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.01)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    # paramwise_cfg=dict(custom_keys=custom_keys, norm_decay_mult=0.0)
)

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

