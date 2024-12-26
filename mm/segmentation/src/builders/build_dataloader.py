
from mmengine.config import Config
from torch.utils.data import DataLoader
import copy
from mmseg.registry import DATASETS
import mm.segmentation.src

def build_dataloader(dataloader):
    if isinstance(dataloader, DataLoader):
        return dataloader

    dataloader_cfg = copy.deepcopy(dataloader)

    # build dataset
    dataset_cfg = dataloader_cfg.pop('dataset')
    if isinstance(dataset_cfg, dict):
        dataset = DATASETS.build(dataset_cfg)
        if hasattr(dataset, 'full_init'):
            dataset.full_init()
    else:
        # fallback to raise error in dataloader
        # if `dataset_cfg` is not a valid type
        dataset = dataset_cfg

    num_batch_per_epoch = dataloader_cfg.pop('num_batch_per_epoch', None)
    if num_batch_per_epoch is not None:
        world_size = get_world_size()
        num_samples = (
            num_batch_per_epoch * _get_batch_size(dataloader_cfg) *
            world_size)
        dataset = _SlicedDataset(dataset, num_samples)

    # build sampler
    sampler_cfg = dataloader_cfg.pop('sampler')
    if isinstance(sampler_cfg, dict):
        sampler_seed = None if diff_rank_seed else seed
        sampler = DATA_SAMPLERS.build(
            sampler_cfg,
            default_args=dict(dataset=dataset, seed=sampler_seed))
    else:
        # fallback to raise error in dataloader
        # if `sampler_cfg` is not a valid type
        sampler = sampler_cfg

    # build batch sampler
    batch_sampler_cfg = dataloader_cfg.pop('batch_sampler', None)
    if batch_sampler_cfg is None:
        batch_sampler = None
    elif isinstance(batch_sampler_cfg, dict):
        batch_sampler = DATA_SAMPLERS.build(
            batch_sampler_cfg,
            default_args=dict(
                sampler=sampler,
                batch_size=dataloader_cfg.pop('batch_size')))
    else:
        # fallback to raise error in dataloader
        # if `batch_sampler_cfg` is not a valid type
        batch_sampler = batch_sampler_cfg

    # build dataloader
    init_fn: Optional[partial]

    if 'worker_init_fn' in dataloader_cfg:
        worker_init_fn_cfg = dataloader_cfg.pop('worker_init_fn')
        worker_init_fn_type = worker_init_fn_cfg.pop('type')
        if isinstance(worker_init_fn_type, str):
            worker_init_fn = FUNCTIONS.get(worker_init_fn_type)
        elif callable(worker_init_fn_type):
            worker_init_fn = worker_init_fn_type
        else:
            raise TypeError(
                'type of worker_init_fn should be string or callable '
                f'object, but got {type(worker_init_fn_type)}')
        assert callable(worker_init_fn)
        init_fn = partial(worker_init_fn,
                            **worker_init_fn_cfg)  # type: ignore
    else:
        if seed is not None:
            disable_subprocess_warning = dataloader_cfg.pop(
                'disable_subprocess_warning', False)
            assert isinstance(disable_subprocess_warning, bool), (
                'disable_subprocess_warning should be a bool, but got '
                f'{type(disable_subprocess_warning)}')
            init_fn = partial(
                default_worker_init_fn,
                num_workers=dataloader_cfg.get('num_workers'),
                rank=get_rank(),
                seed=seed,
                disable_subprocess_warning=disable_subprocess_warning)
        else:
            init_fn = None

    # `persistent_workers` requires pytorch version >= 1.7
    if ('persistent_workers' in dataloader_cfg
            and digit_version(TORCH_VERSION) < digit_version('1.7.0')):
        print_log(
            '`persistent_workers` is only available when '
            'pytorch version >= 1.7',
            logger='current',
            level=logging.WARNING)
        dataloader_cfg.pop('persistent_workers')

    # The default behavior of `collat_fn` in dataloader is to
    # merge a list of samples to form a mini-batch of Tensor(s).
    # However, in mmengine, if `collate_fn` is not defined in
    # dataloader_cfg, `pseudo_collate` will only convert the list of
    # samples into a dict without stacking the batch tensor.
    collate_fn_cfg = dataloader_cfg.pop('collate_fn',
                                        dict(type='pseudo_collate'))
    if isinstance(collate_fn_cfg, dict):
        collate_fn_type = collate_fn_cfg.pop('type')
        if isinstance(collate_fn_type, str):
            collate_fn = FUNCTIONS.get(collate_fn_type)
        else:
            collate_fn = collate_fn_type
        collate_fn = partial(collate_fn, **collate_fn_cfg)  # type: ignore
    elif callable(collate_fn_cfg):
        collate_fn = collate_fn_cfg
    else:
        raise TypeError(
            'collate_fn should be a dict or callable object, but got '
            f'{collate_fn_cfg}')
    data_loader = DataLoader(
        dataset=dataset,
        sampler=sampler if batch_sampler is None else None,
        batch_sampler=batch_sampler,
        collate_fn=collate_fn,
        worker_init_fn=init_fn,
        **dataloader_cfg)
    return data_loader

if __name__ == '__main__':
    from pathlib import Path 
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]

    cfg = Config.fromfile(ROOT / 'configs/_base_/datasets/mask.py')
    
    
    # set crop-size/model-size =================================================================================
    height = 640
    width = 640
    new_crop_size = (height, width)
    cfg.crop_size = new_crop_size 
    if 'train_pipeline' in cfg and isinstance(cfg.train_pipeline, list):
        for pipeline in cfg.train_pipeline:
            if pipeline.get('type') == 'RandomCrop':
                pipeline['crop_size'] = tuple(new_crop_size)
                
                
            if cfg.dataset_type == 'MaskDataset':
                if pipeline.get('type') == 'LoadAnnotations':
                    pipeline['reduce_zero_label'] = True

    if 'test_pipeline' in cfg and isinstance(cfg.test_pipeline, list):
        for pipeline in cfg.test_pipeline:
            if pipeline.get('type') == 'RandomCrop':
                pipeline['crop_size'] = tuple(new_crop_size)
                
            if cfg.dataset_type == 'MaskDataset':
                if pipeline.get('type') == 'LoadAnnotations':
                    pipeline['reduce_zero_label'] = True

    cfg.train_dataloader.dataset.pipeline = cfg.train_pipeline
    cfg.val_dataloader.dataset.pipeline = cfg.test_pipeline

    
    
    data_root = "/HDD/datasets/projects/LX/24.12.12/split_mask_patch_dataset"
    img_suffix='.png'
    seg_map_suffix='.png'
    # classes=('background', 'timber', 'screw')
    classes=('timber', 'screw')
    batch_size = 1
    # set dataset ====================================================================================================
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.dataset['data_root'] = data_root
    cfg.train_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
    cfg.train_dataloader.dataset['classes'] = classes
    cfg.train_dataloader.dataset['img_suffix'] = img_suffix
    
    cfg.val_dataloader.batch_size = batch_size
    cfg.val_dataloader.dataset['data_root'] = data_root
    cfg.val_dataloader.dataset['classes'] = classes
    cfg.val_dataloader.dataset['img_suffix'] = img_suffix
    cfg.val_dataloader.dataset['seg_map_suffix'] = seg_map_suffix
    
    # dataloader = build_dataloader(cfg.train_dataloader)
    dataset = DATASETS.build(cfg.train_dataloader.dataset)

