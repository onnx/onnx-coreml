from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.testing as npt
from onnx import helper, TensorProto
import onnx_caffe2.backend

from onnx_coreml import convert


def _onnx_create_model(nodes, inputs, outputs, initializer=[]):
    initializer_inputs = [
        helper.make_tensor_value_info(
            t.name,
            TensorProto.FLOAT,
            t.dims
        ) for t in initializer
    ]

    graph = helper.make_graph(
        nodes=nodes,
        name="test",
        inputs=initializer_inputs + [
            helper.make_tensor_value_info(
                input_[0],
                TensorProto.FLOAT,
                input_[1]
            ) for input_ in inputs
        ],
        outputs=[
            helper.make_tensor_value_info(
                output_[0],
                TensorProto.FLOAT,
                output_[1]
            ) for output_ in outputs
        ],
        initializer=initializer
    )
    onnx_model = helper.make_model(graph)
    return onnx_model


def _onnx_create_single_node_model(op_type, input_shapes, output_shapes,
                                   initializer=[], **kwargs):
    inputs = [
        ("input{}".format(i,), input_shapes[i])
        for i in range(len(input_shapes))
    ]
    outputs = [
        ("output{}".format(i,), output_shapes[i])
        for i in range(len(output_shapes))
    ]

    node = helper.make_node(
        op_type,
        inputs=[i[0] for i in inputs] + [t.name for t in initializer],
        outputs=[o[0] for o in outputs],
        **kwargs
    )
    return _onnx_create_model([node], inputs, outputs, initializer)


def _shape_from_onnx_value_info(v):
    return tuple([d.dim_value for d in v.type.tensor_type.shape.dim])


def _forward_onnx_model(model, input_dict):
    prepared_backend = onnx_caffe2.backend.prepare(model)
    out = prepared_backend.run(input_dict)
    result = [out[v.name] for v in model.graph.output]
    output_shapes = [
        _shape_from_onnx_value_info(o) for o in model.graph.output
    ]
    for i, output in enumerate(result):
        result[i] = output.reshape(output_shapes[i])
    return np.array(result)


def _coreml_forward_model(model, input_dict, output_names):
    for k, arr in input_dict.items():
        if len(arr.shape) == 4:
            input_dict[k] = arr[0]
    coreml_out = model.predict(input_dict, useCPUOnly=True)
    return np.array([coreml_out[name] for name in output_names])


def _coreml_forward_onnx_model(model, input_dict):
    coreml_model = convert(model)
    output_names = [o.name for o in model.graph.output]
    return _coreml_forward_model(coreml_model, input_dict, output_names)


def _random_array(shape):
    return np.random.ranf(shape).astype("float32")


def _conv_pool_output_size(input_shape, dilations,
                           kernel_shape, pads, strides):
    output_height = (
        input_shape[2] + pads[0] + pads[2] -
        (dilations[0] * (kernel_shape[0] - 1) + 1)
    ) / strides[0] + 1
    output_width = (
        input_shape[3] + pads[1] + pads[3] -
        (dilations[1] * (kernel_shape[1] - 1) + 1)
    ) / strides[1] + 1

    return (int(output_height), int(output_width))


def _assert_outputs(output1, output2, decimal=7):
    npt.assert_equal(len(output1), len(output2))
    for o1, o2 in zip(output1, output2):
        npt.assert_equal(o1.shape, o2.shape)
        npt.assert_almost_equal(
            o2,
            o1,
            decimal=decimal
        )


def _prepare_inputs_for_onnx(model, values=None):
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}
    input_names = [
        i.name for i in graph.input if i.name not in initializer_names
    ]
    input_shapes = [
        tuple([d.dim_value for d in i.type.tensor_type.shape.dim])
        for i in graph.input if i.name not in initializer_names
    ]

    if values is None:
        inputs = [_random_array(shape) for shape in input_shapes]
    else:
        inputs = values

    return dict(zip(input_names, inputs))


def _test_onnx_model(model, decimal=5):
    W = _prepare_inputs_for_onnx(model)
    c2_outputs = _forward_onnx_model(model, W)
    coreml_outputs = _coreml_forward_onnx_model(model, W)
    _assert_outputs(c2_outputs, coreml_outputs, decimal=decimal)


def _test_single_node(op_type, input_shapes, output_shapes,
                      initializer=[], decimal=5, **kwargs):
    model = _onnx_create_single_node_model(
        op_type, input_shapes, output_shapes, initializer, **kwargs
    )
    _test_onnx_model(model, decimal=decimal)
