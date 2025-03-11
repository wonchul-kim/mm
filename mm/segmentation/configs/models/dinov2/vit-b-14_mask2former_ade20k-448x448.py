_base_ = [
        '../../_base_/default_runtime.py', 
        '../../_base_/datasets/ade20k-448x448.py',
        '../../_base_/models/dinov2/vit-s-14_mask2former_ade20k-448x448.py',
]

crop_size = (448, 448)
num_classes = 150
data_preprocessor=dict(
        type="SegDataPreProcessor",
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        size=crop_size,
        bgr_to_rgb=True,
        pad_val=0,
        seg_pad_val=255,
    )
model = dict(
    type="EncoderDecoder",
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="DinoVisionTransformer",
        patch_size=14,
        embed_dim=768,
        depth=12,
        out_indices=[2, 5, 8, 11],
        num_heads=12,
        img_size=crop_size,
        init_cfg=dict(
            type="Pretrained",
            checkpoint="/HDD/weights/dinov2/dinov2_vitb14_pretrain.pth",
        ),
    ),
    decode_head=dict(
        type="Mask2FormerHead",
        in_channels=[768, 768, 768, 768],
        num_classes=num_classes,
        loss_cls=dict(
            type="mmdet.CrossEntropyLoss",
            use_sigmoid=False,
            loss_weight=2.0,
            reduction="mean",
            class_weight=[1.0] * num_classes + [0.1],
        ),
    ),
)


max_epochs = 1000
embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    optimizer=dict(
        type='AdamW',
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-08,
        betas=(0.9, 0.999)),
    paramwise_cfg=dict(
        custom_keys=dict(
            norm=dict(decay_mult=0.0),
            query_embed=dict(lr_mult=1.0, decay_mult=0.0),
            query_feat=dict(lr_mult=1.0, decay_mult=0.0),
            level_embed=dict(lr_mult=1.0, decay_mult=0.0),
            backbone=dict(lr_mult=0.1)),
        norm_decay_mult=0.0)
)
param_scheduler = [
    dict(type="PolyLR", eta_min=0, power=0.9, begin=0, end=40000, by_epoch=False)
]


default_hooks = dict(
    checkpoint=dict(by_epoch=False, interval=20, type='CheckpointHook'),
    logger=dict(interval=100, log_metric_by_epoch=False, type='LoggerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    timer=dict(type='IterTimerHook'),
    visualization=dict(type='SegVisualizationHook', draw=True, interval=100))


train_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     keep_results=True, classwise=True)
val_evaluator = train_evaluator
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     keep_results=True, classwise=True)

train_cfg = dict(type="IterBasedTrainLoop", max_iters=40000, val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')