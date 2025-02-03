# codebase_config = dict(type='mmseg', task='Segmentation', with_argmax=True)

# backend_config = dict(
#     type='tensorrt', common_config=dict(fp16_mode=False, max_workspace_size=0))


# onnx_config = dict(
#     type='onnx',
#     export_params=True,
#     keep_initializers_as_inputs=False,
#     opset_version=13,
#     save_file='end2end.onnx',
#     input_names=['data'],
#     output_names=['output'],
#     input_shape=None,
#     optimize=True)

# onnx_config = dict(input_shape=[1120, 768])
# backend_config = dict(
#     common_config=dict(max_workspace_size=1 << 30),
#     model_inputs=[
#         dict(
#             input_shapes=dict(
#                 input=dict(
#                     min_shape=[1, 3, 768, 1120],
#                     opt_shape=[1, 3, 768, 1120],
#                     max_shape=[1, 3, 768, 1120])))
#     ])

# codebase_config = dict(with_argmax=False)

codebase_config = dict(type='mmseg', task='Segmentation', with_argmax=True)

onnx_config = dict(
    type='onnx',
    export_params=True,
    keep_initializers_as_inputs=False,
    opset_version=None,
    save_file='ONNX.onnx',
    input_names=None,
    output_names=None,
    input_shape=None,
    optimize=True)

backend_config = dict(
    type='tensorrt',
    common_config=dict(max_workspace_size=1 << 30, fp16_mode=False),
    model_inputs=[
        dict(
            input_shapes=dict(
                input=dict(
                    min_shape=[1, 3, None, None],
                    opt_shape=[1, 3, None, None],
                    max_shape=[1, 3, None, None])))
    ])

