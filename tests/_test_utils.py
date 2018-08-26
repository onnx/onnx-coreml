from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import numpy.testing as npt  # type: ignore
import numpy.random as npr
from onnx import helper, TensorProto, ModelProto, ValueInfoProto, TensorProto, NodeProto
from typing import Any, Sequence, Text, Tuple, Optional, Dict, List, TypeVar
from onnx_coreml import convert
from onnx_coreml._graph import Node
import sys
import shutil
import os

'''
0: dynamically generate random inputs, 
   use caffe2 backend for onnx and 
   also save out the generated input and output dicts for future use 
1: use the already saved out input and output dicts for testing
'''
TEST_MODE = 1

def _forward_onnx_model(model,  # type: ModelProto
                        input_dict,  # type: Dict[Text, np._ArrayLike[Any]]
                        test_name = '' # type: Text
                        ):
    # type: (...) -> np.ndarray[Any]
    if TEST_MODE:
        current_dir_path = os.path.dirname(os.path.realpath(__file__))
        loaded_obj = np.load(current_dir_path + '/test_data/' + test_name + '/output.npy', encoding='bytes') #type: ignore
        out = loaded_obj.item()
    else:
        import caffe2.python.onnx.backend # type: ignore
        prepared_backend = caffe2.python.onnx.backend.prepare(model)
        out = prepared_backend.run(input_dict)
        out_dict = {}
        out_names = [v.name for v in model.graph.output]
        for out_name in out_names:
            out_dict[out_name] = out[out_name]
        dir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/' + test_name + '/'
        np.save(dir + 'output.npy', out_dict) #type: ignore

    result = [out[v.name] for v in model.graph.output]
    output_shapes = [
        _shape_from_onnx_value_info(o) for o in model.graph.output
    ]
    for i, output in enumerate(result):
        result[i] = output.reshape(output_shapes[i])
    return np.array(result)


def _onnx_create_model(nodes,  # type: Sequence[NodeProto]
                       inputs,  # type: Sequence[Tuple[Text,Tuple[int, ...]]]
                       outputs,  # type: Sequence[Tuple[Text,Tuple[int, ...], int]]
                       initializer=[],  # type: Sequence[TensorProto]
                       ):
    # type: (...) -> ModelProto
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
                output_[2],
                output_[1]
            ) for output_ in outputs
        ],
        initializer=initializer
    )
    onnx_model = helper.make_model(graph)
    return onnx_model


def _onnx_create_single_node_model(op_type,  # type: Text
                                   input_shapes,  # type: Sequence[Tuple[int, ...]]
                                   output_shapes,  # type: Sequence[Tuple[int, ...]]
                                   initializer=[],  # type: Sequence[TensorProto]
                                   **kwargs  # type: Any
                                   ):
    # type: (...) -> ModelProto
    inputs = [
        ("input{}".format(i,), input_shapes[i])
        for i in range(len(input_shapes))
    ]
    outputs = [
        ("output{}".format(i,), output_shapes[i], TensorProto.FLOAT)
        for i in range(len(output_shapes))
    ]

    node = helper.make_node(
        op_type,
        inputs=[i[0] for i in inputs] + [t.name for t in initializer],
        outputs=[o[0] for o in outputs],
        **kwargs
    )
    return _onnx_create_model([node], inputs, outputs, initializer)


def _shape_from_onnx_value_info(v):  # type: (ValueInfoProto) -> Sequence[Tuple[int, ...]]
    return tuple([d.dim_value for d in v.type.tensor_type.shape.dim])

def _coreml_forward_model(model,  # type: ModelProto
                          input_dict,  # type: Dict[Text, np._ArrayLike[Any]]
                          output_names,  # type: Sequence[Text]
                          ):
    # type: (...) -> np.ndarray[Any]
    for k, arr in input_dict.items():
        if len(arr.shape) == 4:
            input_dict[k] = arr[0]
    coreml_out = model.predict(input_dict, useCPUOnly=True)
    return np.array([coreml_out[name] for name in output_names])


def _coreml_forward_onnx_model(model,  # type: ModelProto
                               input_dict,  # type: Dict[Text, np._ArrayLike[Any]]
                               ):
    # type: (...) -> np.ndarray[Any]
    coreml_model = convert(model)
    output_names = [o.name for o in model.graph.output]
    return _coreml_forward_model(coreml_model, input_dict, output_names)


def _random_array(shape, random_seed=10):  # type: (Tuple[int, ...], Any) -> np._ArrayLike[float]
    if random_seed:
        npr.seed(random_seed) # type: ignore
    return npr.ranf(shape).astype("float32")


def _conv_pool_output_size(input_shape,  # type: Sequence[int]
                           dilations,  # type: Sequence[int]
                           kernel_shape,  # type: Tuple[int, int]
                           pads,  # type: Sequence[int]
                           strides,  # type: Tuple[int, int]
                           ):
    # type: (...) -> Tuple[int, int]
    output_height = (
        input_shape[2] + pads[0] + pads[2] -
        (dilations[0] * (kernel_shape[0] - 1) + 1)
    ) / strides[0] + 1
    output_width = (
        input_shape[3] + pads[1] + pads[3] -
        (dilations[1] * (kernel_shape[1] - 1) + 1)
    ) / strides[1] + 1

    return (int(output_height), int(output_width))


_T = TypeVar('_T')


def _assert_outputs(output1,  # type: np.ndarray[_T]
                    output2,  # type: np.ndarray[_T]
                    decimal=7,  # type: int
                    ):
    # type: (...) -> None
    npt.assert_equal(len(output1), len(output2))
    for o1, o2 in zip(output1, output2):
        npt.assert_almost_equal(
            o2.flatten(),
            o1.flatten(),
            decimal=decimal
        )


def _prepare_inputs_for_onnx(model,  # type: ModelProto
                             test_name = '', # type: Text
                             values=None,  # type: Optional[List[np._ArrayLike[Any]]]
                             ):
    # type: (...) -> Dict[Text, np._ArrayLike[Any]]
    graph = model.graph
    initializer_names = {t.name for t in graph.initializer}
    input_names = [
        i.name for i in graph.input if i.name not in initializer_names
    ]
    input_shapes = [
        tuple([d.dim_value for d in i.type.tensor_type.shape.dim])
        for i in graph.input if i.name not in initializer_names
    ]

    if TEST_MODE:
        dir_path = os.path.dirname(os.path.realpath(__file__))
        loaded_obj = np.load(dir_path + '/test_data/' + test_name + '/input.npy', encoding='bytes') # type: ignore
        return loaded_obj.item() # type: ignore
    else:
        if values is None:
            inputs = [_random_array(shape) for shape in input_shapes]
        else:
            inputs = values
        input_dict = dict(zip(input_names, inputs))
        dir = os.path.dirname(os.path.realpath(__file__)) + '/test_data/' + test_name + '/'
        if os.path.exists(dir):
            shutil.rmtree(dir)
        os.makedirs(dir)
        np.save(dir + 'input.npy', input_dict) # type: ignore
        return input_dict


def _test_onnx_model(model,  # type: ModelProto
                     test_name='', # type: Text
                     decimal=5  # type: int
                     ):
    # type: (...) -> None
    if not test_name:
        test_name = sys._getframe(1).f_code.co_name
    W = _prepare_inputs_for_onnx(model, test_name=test_name)
    c2_outputs = _forward_onnx_model(model, W, test_name=test_name)
    coreml_outputs = _coreml_forward_onnx_model(model, W)
    _assert_outputs(c2_outputs, coreml_outputs, decimal=decimal)


def _test_single_node(op_type,  # type: Text
                      input_shapes,  # type: Sequence[Tuple[int, ...]]
                      output_shapes,  # type: Sequence[Tuple[int, ...]]
                      initializer=[],  # type: Sequence[TensorProto]
                      decimal=5,  # type: int
                      test_name = '', # type: Text
                      **kwargs  # type: Any
                      ):
    # type: (...) -> None
    model = _onnx_create_single_node_model(
        op_type, input_shapes, output_shapes, initializer, **kwargs
    )
    if not test_name:
        test_name = sys._getframe(1).f_code.co_name
    _test_onnx_model(model, test_name=test_name, decimal=decimal)
