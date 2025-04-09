
dataset_type = 'YOLOv5CocoDataset'

data_root = ''  # Root path of data
# Path of train annotation file
train_ann_file = 'train.json'
train_data_prefix = 'train'  # Prefix of train image path
# Path of val annotation file
val_ann_file = 'val.json'
val_data_prefix = 'val'  # Prefix of val image path

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

