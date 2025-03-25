
dataset_type = 'YOLOv5LabelmeDataset'

data_root = None
classes = None
num_classes = None  # Number of classes for classification
img_scale = (0, 0)  # width, height

train_batch_size_per_gpu = None # Batch size of a single GPU during training
train_num_workers = 8 # Worker to pre-fetch data for each single GPU during training

val_batch_size_per_gpu = None # Batch size of a single GPU during validation
val_num_workers = 2 # Worker to pre-fetch data for each single GPU during validation

# Config of batch shapes. Only on val.
# We tested YOLOv8-m will get 0.02 higher than not using it.
batch_shapes_cfg = None
# You can turn on `batch_shapes_cfg` by uncommenting the following lines.
# batch_shapes_cfg = dict(
#     type='BatchShapePolicy',
#     batch_size=val_batch_size_per_gpu,
#     img_size=img_scale[0],
#     # The image scale of padding should be divided by pad_size_divisor
#     size_divisor=32,
#     # Additional paddings for pixel scale
#     extra_pad_ratio=0.5)


# YOLOv5RandomAffine aspect ratio of width and height thres to filter bboxes
max_aspect_ratio = 100
affine_scale = 0.5  # YOLOv5RandomAffine scaling ratio

# persistent_workers must be False if num_workers is 0
persistent_workers = True

albu_train_transforms = [
    dict(type='Blur', p=0.01),
    dict(type='MedianBlur', p=0.01),
    dict(type='ToGray', p=0.01),
    dict(type='CLAHE', p=0.01)
]

pre_transform = [
    # dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True)
]

last_transform = [
    dict(
        type='mmdet.Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags']),
        keymap={
            'img': 'image',
            'gt_bboxes': 'bboxes'
        }),
    dict(type='YOLOv5HSVRandomAug'),
    dict(type='mmdet.RandomFlip', prob=0.5),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'flip',
                   'flip_direction'))
]

train_pipeline = [
    *pre_transform,
    dict(
        type='Mosaic',
        img_scale=img_scale,
        pad_val=114.0,
        pre_transform=pre_transform),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        # img_scale is (width, height)
        border=(-img_scale[0] // 2, -img_scale[1] // 2),
        border_val=(114, 114, 114)),
    *last_transform
]

train_pipeline_stage2 = [
    *pre_transform,
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=True,
        pad_val=dict(img=114.0)),
    dict(
        type='YOLOv5RandomAffine',
        max_rotate_degree=0.0,
        max_shear_degree=0.0,
        scaling_ratio_range=(1 - affine_scale, 1 + affine_scale),
        max_aspect_ratio=max_aspect_ratio,
        border_val=(114, 114, 114)), *last_transform
]

train_dataloader = dict(
    batch_size=train_batch_size_per_gpu,
    num_workers=train_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    collate_fn=dict(type='yolov5_collate'),
    dataset=dict( ############################# Dataset ----------------
        type=dataset_type,
        mode='train',
        data_root=data_root,
        data_prefix=dict(img='train'),
        classes=classes,
        rois=None,
        img_suffix='.bmp',
        ann_suffix='.json',
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        pipeline=train_pipeline,
    )
)

test_pipeline = [
    # dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadImageFromFile', backend_args=None),
    dict(type='YOLOv5KeepRatioResize', scale=img_scale),
    dict(
        type='LetterResize',
        scale=img_scale,
        allow_scale_up=False,
        pad_val=dict(img=114)),
    dict(type='LoadAnnotations', with_bbox=True, _scope_='mmdet'),
    dict(
        type='mmdet.PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'pad_param'))
]

val_dataloader = dict(
    batch_size=val_batch_size_per_gpu,
    num_workers=val_num_workers,
    persistent_workers=persistent_workers,
    pin_memory=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict( ############################# Dataset ----------------
        type=dataset_type,
        mode='val',
        data_root=data_root,
        classes=classes,
        rois=None, 
        img_suffix='.bmp',
        ann_suffix='.json',
        test_mode=True,
        data_prefix=dict(img='val'),
        pipeline=test_pipeline,
        batch_shapes_cfg=batch_shapes_cfg))

test_dataloader = val_dataloader

# FIXME
val_evaluator = dict(
    type='mmdet.CocoMetric',
    proposal_nums=(100, 1, 10),
    ann_file=data_root,
    metric='bbox')
test_evaluator = val_evaluator

