from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import copy

from typing import Sequence, Callable, List, Tuple, Optional, Text, Any
from coremltools.models.neural_network import NeuralNetworkBuilder  #type: ignore
from ._graph import Node, Graph
from coremltools.proto import NeuralNetwork_pb2 #type: ignore
from ._error_utils import ErrorHandling

from ._operators import _convert_relu

INT_MAX = 2**30


def _convert_matmul(builder, node, graph, err):
    '''
    convert to CoreML BatchedMatMul Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L3473
    '''

    weight_name = node.inputs[1]
    W = None
    weight_as_layer_parameter = False
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]

    if W is not None:
        if len(W.shape) != 2:
            # since weight as parameter in batchedMatMul layer must be rank 2
            builder.add_load_constant_nd(node.name + '_const_weight_input', weight_name, constant_value=W,shape=W.shape)
        else:
            weight_as_layer_parameter = True

    if weight_as_layer_parameter:
        builder.add_batched_mat_mul(name=node.name,
                                    input_names=[node.inputs[0]],
                                    output_name=node.outputs[0],
                                    weight_matrix_rows=W.shape[0],
                                    weight_matrix_columns=W.shape[1],
                                    W=W)
    else:
        builder.add_batched_mat_mul(name=node.name,
                                    input_names=[node.inputs[0], weight_name],
                                    output_name=node.outputs[0])


_ONNX_NODE_REGISTRY_ND = {
    "MatMul": _convert_matmul,
    "Relu": _convert_relu,
}

def _get_node_converter_fn(builder, node, err):  # type: (NeuralNetworkBuilder, Node, ErrorHandling) -> Callable[[NeuralNetworkBuilder, Node, Graph, ErrorHandling], None]
    """
    Get the right converter function for ONNX node op_type
    """
    op_type = node.op_type
    if op_type in _ONNX_NODE_REGISTRY_ND:
        return _ONNX_NODE_REGISTRY_ND[op_type]
    else:
        return err.unsupported_op(node)

def _convert_node_nd(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    converter_fn = _get_node_converter_fn(builder, node, err)
    return converter_fn(builder, node, graph, err)

