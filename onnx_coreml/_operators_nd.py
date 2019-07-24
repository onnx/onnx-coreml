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

def _convert_concat(builder, node, graph, err):
    '''
    convert to CoreML ConcatND Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L3521
    '''

    axis = node.attrs.get('axis')
    for i in range(len(node.inputs)):
        if node.inputs[i] in node.input_tensors and node.inputs[i] not in graph.constants_loaded:
            builder.add_load_constant_nd(
                name=node.name + '_load_constant_' + str(i),
                output_name=node.inputs[i],
                constant_value=node.input_tensors[node.inputs[i]],
                shape=node.input_tensors[node.inputs[i]].shape
            )
            graph.constants_loaded.add(node.inputs[i])

    builder.add_concat_nd(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        axis=axis
    )

def _convert_constant(builder, node, graph, err):
    '''
    convert to CoreML Load Constant ND Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L3596
    '''

    value = node.attrs['value']
    # HACK: If Value is 0-Rank then make it 1-Rank
    builder.add_load_constant_nd(
        name=node.name,
        output_name=node.outputs[0],
        constant_value=value,
        shape=[1] if value.shape == () else value.shape
    )
    graph.constants_loaded(node.outputs[0])

def _convert_constant_of_shape(builder, node, graph, err):
    '''
    convert to CoreML Fill Static Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L3641
    '''

    value = node.attrs.get('value', [0.0])
    # if shape is known, create tensor of given shape
    # otherwise create tensor at runtime
    if node.inputs[0] in node.input_tensors:
        builder.add_fill_static(
            name=node.name,
            output_name=node.outputs[0],
            output_shape=node.input_tensors[node.input[0]],
            value=value[0]
        )
    else:
        builder.add_fill_dynamic(
            name=node.name,
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            value=value[0]
        )

def _convert_gather(builder, node, graph, err):
    '''
    convert to CoreML Gather Along Axis Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L4296
    '''
    axis = node.attrs.get('axis', 0)

    if len(node.inputs) != 2:
        err.unsupported_op_configuration(builder, node, graph, "Error in ONNX model: Gather expects two inputs")
    
    data = node.inputs[0]
    indices = node.inputs[1]
    if node.inputs[0] in node.input_tensors and node.inputs[0] not in graph.constants_loaded:
        builder.add_load_constant_nd(
            name=node.name + '_load_data',
            output_name=node.inputs[0],
            constant_value=node.input_tensors[node.inputs[0]],
            shape=node.input_tensors[node.inputs[0]].shape
        )
        graph.constants_loaded.add(node.inputs[0])
    
    if node.inputs[1] in node.input_tensors and node.inputs[1] not in graph.constants_loaded:
        builder.add_load_constant_nd(
            name=node.name+ '_load_indices',
            output_name=node.inputs[1],
            constant_value=node.input_tensors[node.inputs[1]],
            shape=node.input_tensors[node.inputs[1]].shape
        )
        graph.constants_loaded.add(node.inputs[1])
    
    builder.add_gather(
        name=node.name,
        input_names=[data, indices],
        output_name=node.outputs[0],
        axis=axis
    )

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


def _convert_reshape(builder, node, graph, err):
    '''
    convert to CoreML Reshape Static Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L4844
    '''
    shape_node = node.inputs[1]
    if shape_node in node.input_tensors:
        output_shape = node.input_tensors[shape_node]
    
        # if rank is same, then call rank preserving reshape
        if node.inputs[0] not in graph.shape_dict:
            err.unsupported_op_configuration(builder, node, graph, "Input shape not represented in graph")
    
        len_of_input_shape = len(graph.shape_dict[node.inputs[0]])
        if len(output_shape) == len_of_input_shape:
            builder.add_rank_preserving_reshape(
                name=node.name,
                input_name=node.inputs[0],
                output_name=node.outputs[0],
                output_shape=output_shape
            )
        else:
            add_static_reshape = True
            if len_of_input_shape > len(output_shape):
                num_zeros = 0
                num_neg_ones = 0
                for i in output_shape:
                    if i == 0:
                        num_zeros += 1
                    elif i == -1:
                        num_neg_ones += 1

                if num_neg_ones > 1:
                     err.unsupported_op_configuration(builder, node, graph, "Error in ONNX model: At most one dimension of new shape can be -1, found {}".format(num_neg_ones))

                if num_neg_ones + num_zeros == len(output_shape):
                    # Rank of output is less than input
                    # Make Rank equivalent for reshape and then squeeze
                    add_static_reshape = False
                    new_shape = []
                    i = 0
                    for i in range(len(output_shape)):
                        new_shape.append(output_shape[i])
                        if output_shape[i] == -1:
                            break
                    while i < len_of_input_shape-1:
                        new_shape.append(1)
                        i += 1

                    builder.add_rank_preserving_reshape(
                        name=node.name + '_reshape_preserving',
                        input_name=node.inputs[0],
                        output_name=node.outputs[0] + '_reshape_dim_preserved',
                        output_shape=new_shape
                    )

                    squeeze_axes = list(range(len(output_shape) - len_of_input_shape, 0))
                    squeeze_axes.reverse()

                    builder.add_squeeze(
                        name=node.name,
                        input_name=node.outputs[0] + '_reshape_dim_preserved',
                        output_name=node.outputs[0],
                        axes=squeeze_axes
                    )

            if add_static_reshape:    
                builder.add_reshape_static(
                    name=node.name,
                    input_name=node.inputs[0],
                    output_name=node.outputs[0],
                    output_shape=output_shape
                )
    else:
        builder.add_reshape_dynamic(
            name=node.name,
            input_names=node.inputs,
            output_name=node.outputs[0],
        )

def _convert_slice(builder, node, graph, err):
    '''
    convert to CoreML Slice Static Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5082
    ''' 
    
    data_shape = graph.shape_dict[node.inputs[0]]
    len_of_data = len(data_shape)
    begin_masks = [True] * len_of_data
    end_masks = [True] * len_of_data

    default_axes = list(range(len_of_data))
    default_steps = [1] * len_of_data
    
    ip_starts = node.attrs.get('starts')
    ip_ends = node.attrs.get('ends')
    axes = node.attrs.get('axes', default_axes)
    steps = node.attrs.get('steps', default_steps)

    starts = [0] * len_of_data
    ends = [0] * len_of_data

    for i in range(len(axes)):
        current_axes = axes[i]
        starts[current_axes] = ip_starts[i]
        ends[current_axes] = ip_ends[i]
        if ends[current_axes] != INT_MAX or ends[current_axes] < data_shape[current_axes]:
            end_masks[current_axes] = False

        if starts[current_axes] != 0:
            begin_masks[current_axes] = False

    builder.add_slice_static(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        begin_ids=starts,
        end_ids=ends,
        strides=steps,
        begin_masks=begin_masks,
        end_masks=end_masks
    )

def _convert_split(builder, node, graph, err):
    '''
    convert to CoreML Squeeze Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#5003
    '''

    axis = node.attrs.get('axis', 0)
    builder.add_split_nd(
        name=node.name,
        input_name=node.inputs[0],
        output_names=node.outputs,
        axis=axis
    )

def _convert_squeeze(builder, node, graph, err):
    '''
    convert to CoreML Squeeze Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L4903
    '''
    axes = node.attrs.get('axes', None)
    builder.add_squeeze(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        axes=axes
    )

def _convert_shape(builder, node, graph, err):
    '''
    convert to CoreML GetShape Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L5131
    '''
    builder.add_get_shape(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_transpose(builder, node, graph, err):
    '''
    convert to CoreML Transpose Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L3426
    '''
    
    axes = node.attrs.get('perm', [])
    # If 'perm' not provided, the reverse the dimensions
    if axes == []:
        rank = len(graph.shape_dict[node.inputs[0]])
        axes = list(range(-1, -(rank+1), -1))

    builder.add_transpose(
        name=node.name,
        axes=axes,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_unsqueeze(builder, node, graph, err):
    '''
    convert to CoreML ExpandDim Layer:
    https://github.com/apple/coremltools/blob/655b3be5cc0d42c3c4fa49f0f0e4a93a26b3e492/mlmodel/format/NeuralNetwork.proto#L4810
    '''
    axes = node.attrs.get('axes')
    builder.add_expand_dims(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        axes=axes
    )


_ONNX_NODE_REGISTRY_ND = {
    "Concat": _convert_concat,
    "Constant": _convert_constant,
    "ConstantOfShape": _convert_constant_of_shape,
    "Gather": _convert_gather,
    "MatMul": _convert_matmul,
    "Relu": _convert_relu,
    "Reshape": _convert_reshape,
    "Slice": _convert_slice,
    "Split": _convert_split,
    "Shape": _convert_shape,
    "Squeeze": _convert_squeeze,
    "Transpose": _convert_transpose,
    "Unsqueeze": _convert_unsqueeze
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

