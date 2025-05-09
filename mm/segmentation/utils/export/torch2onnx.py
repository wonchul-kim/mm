import os.path as osp
from typing import Any, Optional, Union, Dict

import torch
import mmengine


def torch2onnx(model_inputs: torch.Tensor,
               work_dir: str,
               save_file: str,
               deploy_cfg: Union[str, mmengine.Config],
               model_cfg: Union[str, mmengine.Config],
               model_checkpoint: Optional[str] = None,
               device: str = 'cuda:0',
               tta: Dict = {}):
    from mmdeploy.apis.core.pipeline_manager import no_mp
    from mmdeploy.utils import (Backend, get_backend, get_dynamic_axes,
                                get_input_shape, get_onnx_config, load_config)
    from mmdeploy.apis.onnx import export

    # load deploy_cfg if necessary
    deploy_cfg, model_cfg = load_config(deploy_cfg, model_cfg)
    mmengine.mkdir_or_exist(osp.abspath(work_dir))

    # input_shape = get_input_shape(deploy_cfg)

    # create model an inputs
    from mmdeploy.apis import build_task_processor
    task_processor = build_task_processor(model_cfg, deploy_cfg, device)

    torch_model = task_processor.build_pytorch_model(model_checkpoint)
    # data, model_inputs = task_processor.create_input(
    #     img,
    #     input_shape,
    #     data_preprocessor=getattr(torch_model, 'data_preprocessor', None))

    # if isinstance(model_inputs, list) and len(model_inputs) == 1:
    #     model_inputs = model_inputs[0]
    # data_samples = data['data_samples']
    # input_metas = {'data_samples': data_samples, 'mode': 'predict'}

    if tta != {} and 'augs' in tta:
        from mm.segmentation.src.models.tta_model import TTASegModel
        torch_model = TTASegModel(torch_model, tta['augs'], shape=(deploy_cfg.onnx_config.input_shape[1], deploy_cfg.onnx_config.input_shape[0]))

    # export to onnx
    context_info = dict()
    context_info['deploy_cfg'] = deploy_cfg
    output_prefix = osp.join(work_dir,
                             osp.splitext(osp.basename(save_file))[0])
    backend = get_backend(deploy_cfg).value

    onnx_cfg = get_onnx_config(deploy_cfg)
    opset_version = onnx_cfg.get('opset_version', 11)

    input_names = onnx_cfg['input_names']
    output_names = onnx_cfg['output_names']
    axis_names = input_names + output_names
    dynamic_axes = get_dynamic_axes(deploy_cfg, axis_names)
    verbose = not onnx_cfg.get('strip_doc_string', True) or onnx_cfg.get(
        'verbose', False)
    keep_initializers_as_inputs = onnx_cfg.get('keep_initializers_as_inputs',
                                               True)
    optimize = onnx_cfg.get('optimize', False)
    if backend == Backend.NCNN.value:
        """NCNN backend needs a precise blob counts, while using onnx optimizer
        will merge duplicate initilizers without reference count."""
        optimize = False
    with no_mp():
        export(
            torch_model,
            model_inputs,
            # input_metas=input_metas,
            output_path_prefix=output_prefix,
            backend=backend,
            input_names=input_names,
            output_names=output_names,
            context_info=context_info,
            opset_version=opset_version,
            dynamic_axes=dynamic_axes,
            verbose=verbose,
            keep_initializers_as_inputs=keep_initializers_as_inputs,
            optimize=optimize)
