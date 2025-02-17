dataset_type = "LabelmeDataset"
data_root = None

width = 640
height = 640
train_pipeline = [
    dict(type='LoadImageFromFileWithRoi'),
    dict(type='Resize', scale=(width, height), keep_ratio=False),
    dict(type='RandomFlip', prob=0.3),
    dict(type='PhotoMetricDistortion'),
    dict(type='PackSegInputs', meta_keys=['img_path', 'seg_map_path', 'ori_shape', 
                                          'img_shape', 'pad_shape', 'scale_factor', 
                                          'flip', 'flip_direction', 'reduce_zero_label', 
                                          'classes', 'roi', 'patch'])
]

val_pipeline = [
    dict(type='LoadImageFromFileWithRoi'),
    dict(type='Resize', scale=(width, height), keep_ratio=False),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs', meta_keys=['img_path', 'seg_map_path', 'ori_shape', 
                                          'img_shape', 'pad_shape', 'scale_factor', 
                                          'flip', 'flip_direction', 'reduce_zero_label', 
                                          'classes', 'roi', 'patch'])
]
test_pipeline = [
    dict(type='LoadImageFromFileWithRoi'),
    # dict(type='Resize', scale=(width, height), keep_ratio=False),
    dict(type='RandomFlip', prob=0.0),
    # add loading annotation after ``Resize`` because ground truth
    # does not need to do resize data transform
    # dict(type='LoadAnnotations', reduce_zero_label=True),
    dict(type='PackSegInputs', meta_keys=['img_path', 'seg_map_path', 'ori_shape', 'img_shape', 
                                          'pad_shape', 'scale_factor', 'flip', 'flip_direction', 
                                          'reduce_zero_label', 
                                          'classes', 
                                          'roi', 'patch', 'do_metric', 'annotate'])
]
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
        rois=None,
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
        rois=None,
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
        rois=None,
        img_suffix='.png',
        seg_map_suffix='.png',
        pipeline=val_pipeline))

val_evaluator = dict(type='IoUMetricV2', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     keep_results=True, classwise=True)
test_evaluator = dict(type='IoUMetric', iou_metrics=['mIoU', 'mDice', 'mFscore'],
                     keep_results=True, classwise=True)

train_cfg = dict(
    type='IterBasedTrainLoopV2', max_iters=160000, val_interval=50, 
    evaluator=val_evaluator)
# train_cfg = dict(
#     type='IterBasedTrainLoopV2', max_iters=160000, val_interval=5000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoopV2')