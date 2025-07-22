_base_ = [
    '../../_base_/models/segformer/segformer_mit-b0.py',
    '../../_base_/default_runtime.py', 
    '../../_base_/datasets/labelme.py'
]
GPU = 1
BATCH_SIZE = 2
# optimizer = dict(type='SGD', lr=0.01 / 8 * GPU * BATCH_SIZE, momentum=0.9, weight_decay=0.0005,
# # optimizer = dict(type='SGD', lr=0.01,  momentum=0.9, weight_decay=0.0005,
#     paramwise_cfg = dict(
#         custom_keys={
#             'head': dict(lr_mult=2.),
#             'att': dict(lr_mult=2.),
#             }))

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        type='AdamW', lr=0.00006, betas=(0.9, 0.999), weight_decay=0.01),
    paramwise_cfg=dict(
        custom_keys={
            'head': dict(lr_mult=2.),
            'att': dict(lr_mult=2.),
            }))

lr_config = dict(policy='poly', power=0.9, 
                 min_lr=0.0001, ### 
                #  min_lr=0.0, ### 
                 by_epoch=False, 
                warmup='linear', warmup_iters=200)

train_dataloader = dict(batch_size=BATCH_SIZE, num_workers=2)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

crop_size = (640, 640)
data_preprocessor = dict(
    size=crop_size,
    type='SegDataPreProcessor',
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=255)
norm_cfg = dict(type='SyncBN', requires_grad=True)
model = dict(
    type='EncoderDecoder',
    data_preprocessor=data_preprocessor,
    pretrained='open-mmlab://resnet50_v1c',
    backbone=dict(
        # type='ResNetV1c',
        # frozen_stages=4,
        # type='ResNetV1cWithBlur',
        type='NyResNet', # FreqMix in resnet backbone
        # blur_type='adafreq',
        # blur_type='adablur',
        # blur_type='blur',
        blur_type='flc', # De-aliasing filter
        # blur_type='none',
        freq_thres=0.25 * 1.4, # Set the cutoff frequency
        # blur_k=1,
        # log_aliasing_ratio=True,

        # with_cp=True,
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        dilations=(1, 1, 1, 1),
        strides=(1, 2, 2, 2),
        norm_cfg=norm_cfg,
        norm_eval=False,
        style='pytorch',
        contract_dilation=True),
    decode_head=dict(
        type='UPerHead',
        # type='UPerHeadFreqMix',
        # upsampling_mode='bilinear',
        # k_lists=[[2, 4, 8], [2, 4, 8], [2, 4, 8], [2, 4, 8]],
        in_channels=[256, 512, 1024, 2048],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=128,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=(crop_size[0] % 2 == 1),
        loss_decode=[
            dict(type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            ]
            ),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=1024,
        in_index=2,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=19,
        norm_cfg=norm_cfg,
        align_corners=(crop_size[0] % 2 == 1),
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    # test_cfg=dict(mode='whole')
    # test_cfg=dict(mode='slide', crop_size=(769, 769), stride=(513, 513))
    test_cfg=dict(mode='slide', crop_size=(768 + (crop_size[0] % 2), 768 + (crop_size[0] % 2)), stride=(512 + (crop_size[0] % 2), 512 + (crop_size[0] % 2)))
    )
