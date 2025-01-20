dataset_type = "LabelmeDataset"
data_root = None

width = 640
height = 640
train_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(
    #     type='RandomResize',
    #     scale=(2048, 512),
    #     ratio_range=(0.5, 2.0),
    #     keep_ratio=True),
    # dict(type='RandomCrop', crop_size=(height, width), cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.3),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs')
]
val_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(width, height), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(width, height), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs')
]
# img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
# tta_pipeline = [
#     dict(type='LoadImageFromFile', backend_args=None),
#     dict(
#         type='TestTimeAug',
#         transforms=[
#             [
#                 dict(type='Resize', scale_factor=r, keep_ratio=True)
#                 for r in img_ratios
#             ],
#             [
#                 dict(type='RandomFlip', prob=0., direction='horizontal'),
#                 dict(type='RandomFlip', prob=1., direction='horizontal')
#             ], [dict(type='LoadAnnotations')], [dict(type='PackSegInputs')]
#         ])
# ]
train_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        mode='train',
        data_root=None,
        data_prefix=dict(img_path='train', seg_map_path='train'),
        classes=None,
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        mode='val',
        data_root=None,
        data_prefix=dict(img_path='val', seg_map_path='val'),
        classes=None,
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=val_pipeline))
test_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        mode='test',
        data_root=None,
        data_prefix=dict(img_path='./', seg_map_path='./'),
        classes=None,
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=val_pipeline))

val_evaluator = dict(type='IoUMetricV2', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     keep_results=True, classwise=True)
test_evaluator = val_evaluator
