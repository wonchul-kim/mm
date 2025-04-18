# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.evaluator import DumpResults
from mmengine.runner import Runner

from mmyolo.registry import RUNNERS
from mmyolo.utils import is_metainfo_lower
from mm.yolo.utils.config import TestConfigManager
from mm.utils.functions import add_params_to_args, increment_path
import mm.yolo.src 
import mm.yolo.utils.hooks

from pathlib import Path 
FILE = Path(__file__).resolve()
ROOT = FILE.parents[2]

# TODO: support fuse_conv_bn
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMYOLO test (and eval) a model')
    # parser.add_argument(
    #     '--json-prefix',
    #     type=str,
    #     help='the prefix of the output json file without perform evaluation, '
    #     'which is useful when you want to format the result to a specific '
    #     'format and submit it to the test server')
    # parser.add_argument(
    #     '--tta',
    #     action='store_true',
    #     help='Whether to use test time augmentation')
    # parser.add_argument(
    #     '--show', action='store_true', help='show prediction results')
    # parser.add_argument(
    #     '--deploy',
    #     action='store_true',
    #     help='Switch model to deployment mode')
    # parser.add_argument(
    #     '--show-dir',
    #     help='directory where painted images will be saved. '
    #     'If specified, it will be automatically saved '
    #     'to the work_dir/timestamp/show_dir')
    # parser.add_argument(
    #     '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()
    add_params_to_args(args, ROOT / 'yolo/configs/recipe/test.yaml')
    # add_params_to_args(args, args.args_filename)
    # add_params_to_args(args, '/HDD/_projects/github/mm/mm/yolo/data/projects/tenneco/test_tenneco_outer_coco_unit.yaml')
    add_params_to_args(args, '/HDD/_projects/github/mm/mm/yolo/data/projects/tenneco/test_tenneco_outer_coco.yaml')

    args.create_output_dirs = True
    
    if args.create_output_dirs:
        from mm.utils.functions import create_output_dirs
        create_output_dirs(args, 'test')
        print(f"CREATED output-dirs: {args.output_dir}")
    
    args.load_from = args.weights
    args.data_root = args.input_dir
    args.num_classes = len(args.classes)

    args.custom_hooks['visualize_test']['output_dir'] = osp.join(args.output_dir, 'vis')

    config_file = ROOT / f'yolo/configs/models/{args.model}/{args.model}_{args.backbone}_mask-refine_syncbn_fast_8xb16_coco.py'
    config_manager = TestConfigManager()
    config_manager.build(args, config_file)
    config_manager.manage_model_config(args.num_classes, args.width, args.height)
    config_manager.manage_coco_dataset_config(args.data_root, args.img_suffix, args.seg_map_suffix, 
                                         args.classes, args.batch_size, args.width, args.height,
                                         args.rois, args.patch)
    config_manager.manage_custom_test_hooks_config(args.custom_hooks)

    cfg = config_manager.cfg

    # Reduce the number of repeated compilations and improve
    # training speed.
    setup_cache_size_limit_of_dynamo()

    # if args.show or args.show_dir:
    #     cfg = trigger_visualization_hook(cfg, args)

    # if args.deploy:
    #     cfg.custom_hooks.append(dict(type='SwitchToDeployHook'))

    # # add `format_only` and `outfile_prefix` into cfg
    # if args.json_prefix is not None:
    #     cfg_json = {
    #         'test_evaluator.format_only': True,
    #         'test_evaluator.outfile_prefix': args.json_prefix
    #     }
    #     cfg.merge_from_dict(cfg_json)

    # Determine whether the custom metainfo fields are all lowercase
    is_metainfo_lower(cfg)

    # if args.tta:
    #     assert 'tta_model' in cfg, 'Cannot find ``tta_model`` in config.' \
    #                                " Can't use tta !"
    #     assert 'tta_pipeline' in cfg, 'Cannot find ``tta_pipeline`` ' \
    #                                   "in config. Can't use tta !"

    #     cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
    #     test_data_cfg = cfg.test_dataloader.dataset
    #     while 'dataset' in test_data_cfg:
    #         test_data_cfg = test_data_cfg['dataset']

    #     # batch_shapes_cfg will force control the size of the output image,
    #     # it is not compatible with tta.
    #     if 'batch_shapes_cfg' in test_data_cfg:
    #         test_data_cfg.batch_shapes_cfg = None
    #     test_data_cfg.pipeline = cfg.tta_pipeline

    # build the runner from config
    if 'runner_type' not in cfg:
        # build the default runner
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)


    # start testing
    runner.test()


if __name__ == '__main__':
    main()