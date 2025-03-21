_base_ = [
        '../../_base_/default_runtime.py', 
        '../../_base_/datasets/labelme.py'
]

crop_size = (1024, 1024)
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
        type='SAM2Encoder',
        model_cfg="configs/sam2.1/sam2.1_hiera_l.yaml",
        exclude_layers = ['sam_mask_decoder', 'sam_prompt_encoder', 'memory_encoder',
                        'memory_attention', 'mask_downsample', 'obj_ptr_tpos_proj', 
                        'obj_ptr_proj', 'image_encoder.neck'],
        checkpoint_path=None,
        frozen_stages=1,
    ),
    decode_head=dict(
        type='SAM2UNetHead',
        num_classes=num_classes,
        rfb_channels = [[144, 64], [288, 64], [576, 64], [1152, 64]],
        norm_cfg=norm_cfg,
        act_cfg=dict(type='ReLU', inplace=True),
        loss_decode=[
            # dict(
            #     type='OhemCrossEntropy',
            #     thres=0.9,
            #     min_kept=131072,
            #     class_weight=[1.]*num_classes,
            #     loss_weight=0.4),
            dict(
                type='OhemCrossEntropy',
                thres=0.9,
                min_kept=131072,
                class_weight=[1.]*num_classes,
                loss_weight=1.0),
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