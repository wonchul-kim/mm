_base_ = [
    '../../_base_/models/deeplabv3plus/deeplabv3plus_efficientnetv2-b0.py', 
    '../../_base_/datasets/labelme.py',
    '../../_base_/default_runtime.py', 
    # '../../_base_/schedules/schedule_160k.py'
]
crop_size = (512, 512)
num_classes = 150
data_preprocessor = dict(size=crop_size)
model = dict(
    data_preprocessor=data_preprocessor,
    decode_head=dict(num_classes=num_classes),
    auxiliary_head=dict(num_classes=num_classes))