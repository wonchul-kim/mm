_base_ = [
        '../../_base_/default_runtime.py', 
        '../../_base_/datasets/ade20k_640x640.py',
        '../../_base_/models/dinov2/dinov2.py',
]

max_epochs = 1000
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    decode_head=dict(
        type='BNHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        input_transform='resize_concat',
        channels=3072,
        dropout_ratio=0,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
)

optimizer = dict(type='AdamW', lr=0.001, weight_decay=0.0001, betas=(0.9, 0.999))
optim_wrapper = dict(
    clip_grad=None,
    optimizer=optimizer,
    type='OptimWrapper')
param_scheduler = [ # NOTE: this definition is slightly different from the original config
    dict(
        type='PolyLR',
        power=1.0,
        begin=0,
        end=max_epochs,
        eta_min=0.0,
        by_epoch=True,
    )
]

log_processor = dict(by_epoch=True)
default_hooks = dict(
    checkpoint=dict(by_epoch=True, interval=20, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=True, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))


train_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     keep_results=True, classwise=True)
val_evaluator = train_evaluator
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     keep_results=True, classwise=True)

train_cfg = dict(by_epoch=True, max_epochs=max_epochs, val_interval=1)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')