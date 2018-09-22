from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from typing import Sequence, Callable, List, Tuple, Optional, Text, Any
from coremltools.models.neural_network import NeuralNetworkBuilder  #type: ignore
from ._graph import Node, Graph
from coremltools.proto import NeuralNetwork_pb2 #type: ignore
from ._error_utils import ErrorHandling

def _compare(a, b, encoding="utf8"): #type: (Text, Text, Text) -> bool
    if isinstance(a, bytes):
        a = a.decode(encoding)
    if isinstance(b, bytes):
        b = b.decode(encoding)
    return a == b

def _convert_conv(builder, node, graph, err): # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    #get weights for convolution
    weight_name = node.inputs[1]
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name,))

    is_deconv = False
    if node.op_type.endswith("Transpose"):
        is_deconv = True

    if not is_deconv:
        W = W.transpose((2, 3, 1, 0)) # type: ignore
    else:
        W = W.transpose((2, 3, 0, 1)) # type: ignore
    bias = None
    if len(node.inputs) > 2:
        bias = node.input_tensors[node.inputs[2]]

    dilations = node.attrs.get("dilations", [1, 1])
    groups = 1
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]

    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs["strides"]

    out_shape = None
    output_name = node.outputs[0]
    is_post_pad = False

    padding_type = 'valid'
    same_padding_asymmetry_mode = 'BOTTOM_RIGHT_HEAVY'

    if "auto_pad" in node.attrs and \
            not _compare(node.attrs["auto_pad"], 'VALID'):
        padding_type = 'same'
        if _compare(node.attrs["auto_pad"], 'SAME_LOWER'):
            same_padding_asymmetry_mode = 'TOP_LEFT_HEAVY'

    if is_deconv:
        if 'output_shape' in node.attrs:
            out_shape = (node.attrs['output_shape'][-2], node.attrs['output_shape'][-1]) #(Hout, wout)
        elif 'output_padding' in node.attrs:
            post_pads = node.attrs['output_padding']
            if sum(post_pads) != 0:
                t = l = b = r = 0
                if len(post_pads) == 2:
                    b, r = post_pads
                elif len(post_pads) == 4:
                    t, l, b, r = post_pads
                else:
                    return err.unsupported_op_configuration(builder, node, graph, "Supports only length 2 or 4 output padding attribute")
                is_post_pad = True
                output_name += '_conv_tranpose_pre_pad'

    builder.add_convolution(
        name=node.name,
        kernel_channels=W.shape[2],
        output_channels=W.shape[3],
        height=kernel_shape[0],
        width=kernel_shape[1],
        stride_height=strides[0],
        stride_width=strides[1],
        border_mode=padding_type,
        same_padding_asymmetry_mode=same_padding_asymmetry_mode,
        groups=groups,
        W=W,
        b=bias,
        has_bias=bias is not None,
        is_deconv=is_deconv,
        output_shape=out_shape,
        input_name=node.inputs[0],
        output_name=output_name,
        dilation_factors=dilations,
        padding_top=pads[0],
        padding_bottom=pads[2],
        padding_left=pads[1],
        padding_right=pads[3]
    )

    if is_post_pad:
        builder.add_padding(
            name=node.name + '_post_pad', # type: ignore
            left=l,
            right=r,
            top=t,
            bottom=b,
            value=0,
            input_name=output_name,
            output_name=node.outputs[0],
        )

def _convert_relu(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_activation(
        name=node.name,
        non_linearity='RELU',
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_thresholdedrelu(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    alpha = node.attrs.get('alpha', 1.0)
    builder.add_activation(
        name=node.name,
        non_linearity='THRESHOLDEDRELU',
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        params = alpha
    )

def _convert_reshape(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None

    shape = tuple(node.attrs.get('shape', ())) # type: (Tuple[int, ...])

    if len(shape) == 0:
        shape_name = node.inputs[1]
        if shape_name in node.input_tensors:
            shape = tuple(node.input_tensors[shape_name].astype(int)) #type: ignore
        else:
            err.missing_initializer(node,
                                "Shape tensor: {} not found in the graph initializer".format(shape_name, ))

    def get_coreml_target_shape(target_shape):  # type: (Tuple[int, ...]) -> Optional[Tuple[int, ...]]
        if len(target_shape) == 1:  # (D,)
            coreml_shape = (1, target_shape[0], 1, 1)  # type: Optional[Tuple[int, ...]]
        elif len(target_shape) == 2:  # (S,D)
            coreml_shape = target_shape + (1, 1)
        elif len(target_shape) == 3:  # (C,H,W)
            coreml_shape = (
                1, target_shape[0], target_shape[1], target_shape[2]
            )
        elif len(target_shape) == 4:
            coreml_shape = target_shape
        elif len(target_shape) > 4:
            diff = len(shape) - 4
            if all([d == 1 for d in shape[:diff]]):
                coreml_shape = shape[diff:]
            else:
                return err.unsupported_op_configuration(builder, node, graph, "Supports only 3d and 4d tensors") # type: ignore
        else:
            coreml_shape = None
        return coreml_shape

    # check if all entries in shape are 1/-1
    is_flatten = True
    for s in shape:
        if abs(s) != 1:
            is_flatten = False
            break
    if is_flatten:
        builder.add_flatten(
            name=node.name,
            input_name=node.inputs[0],
            output_name=node.outputs[0],
            mode=0
        )
        return

    new_shape = get_coreml_target_shape(shape)

    if new_shape is None:
        return err.unsupported_op_configuration(builder, node, graph, "Unsupported shape for reshape")

    builder.add_reshape(
        name=node.name,
        target_shape=new_shape,
        mode=0,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_transpose(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    perm = node.attrs.get("perm", [0, 3, 2, 1])
    if len(perm) > 4:
        diff = len(perm) - 4
        if all([perm[i] == i for i in range(diff)]):
            perm = [p - diff for p in perm[diff:]]
        else:
            return err.unsupported_op_configuration(builder, node, graph, "Supports only 4d tensors")
    elif len(perm) < 4:
        diff = 4 - len(perm)
        perm = [d for d in range(diff)] + [d + diff for d in perm]
    perm = tuple(perm)

    builder.add_permute(
        name=node.name,
        dim=perm,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_pool(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    is_global = False
    if node.op_type.startswith('Global'):
        is_global = True

    if node.op_type.endswith("MaxPool"):
        layer_type = "MAX"
    elif node.op_type.endswith("AveragePool"):
        layer_type = "AVERAGE"
    else:
        return err.unsupported_op_configuration(builder, node, graph, "Unsupported pool type")

    if len(node.outputs) == 2:
        return err.unsupported_op_configuration(builder, node, graph, "argmax with pool unsupported")

    pad_b, pad_l, pad_r, pad_t = 0, 0, 0, 0
    stride_height, stride_width = 1, 1
    padding_type = 'VALID'
    same_padding_asymmetry_mode = 'BOTTOM_RIGHT_HEAVY'

    if is_global:
        height, width = 0, 0
        stride_height, stride_width = 1, 1
    else:
        kernel_shape = node.attrs["kernel_shape"]
        height = kernel_shape[0]
        width = kernel_shape[1]

        pads = node.attrs.get('pads', [0,0,0,0])
        pad_t = pads[0]
        pad_l = pads[1]
        pad_b = pads[2]
        pad_r = pads[3]

        strides = node.attrs.get('strides', [1,1])
        stride_height = strides[0]
        stride_width = strides[1]

        if "auto_pad" in node.attrs and \
            not _compare(node.attrs["auto_pad"], 'VALID'):
            padding_type = 'SAME'
            if _compare(node.attrs["auto_pad"], 'SAME_LOWER'):
                same_padding_asymmetry_mode = 'TOP_LEFT_HEAVY'

    exclude_pad_area = node.attrs.get('count_include_pad',0) == 0

    builder.add_pooling(
        name=node.name,
        height=height,
        width=width,
        stride_height=stride_height,
        stride_width=stride_width,
        layer_type=layer_type,
        padding_type=padding_type,
        exclude_pad_area=exclude_pad_area,
        is_global=is_global,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        padding_top=pad_t,
        padding_bottom=pad_b,
        padding_left=pad_l,
        padding_right=pad_r,
        same_padding_asymmetry_mode = same_padding_asymmetry_mode
    )

def _convert_fc(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    W = node.input_tensors[node.inputs[1]]
    b = node.input_tensors.get(node.inputs[2])
    output_channels, input_channels = W.shape
    builder.add_inner_product(
        name=node.name,
        W=W,
        b=b,
        input_channels=input_channels,
        output_channels=output_channels,
        has_bias=b is not None,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_bn(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    if len(node.outputs) > 1:
        return err.unsupported_op_configuration(builder, node, graph, "This converter only supports BatchNormalization with one output")

    epsilon = node.attrs.get("epsilon", 1e-5)
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]
    mean = node.input_tensors[node.inputs[3]]
    var = node.input_tensors[node.inputs[4]]

    builder.add_batchnorm(
        name=node.name,
        channels=scale.shape[0],
        gamma=scale,
        beta=bias,
        mean=mean,
        variance=var,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        epsilon=epsilon
    )

def _convert_instancenorm(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    epsilon = node.attrs.get("epsilon", 1e-5)
    scale = node.input_tensors[node.inputs[1]]
    bias = node.input_tensors[node.inputs[2]]

    builder.add_batchnorm(
        name=node.name,
        channels=scale.shape[0],
        gamma=scale,
        beta=bias,
        compute_mean_var=True,
        instance_normalization=True,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        epsilon=epsilon
    )

def _convert_add(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None

    # check if its equivalent to a bias layer
    if len(node.inputs) > 1:
        if node.inputs[1] in node.input_tensors:
            second_input = np.squeeze(node.input_tensors[node.inputs[1]])
            if len(second_input.shape) == 1:
                builder.add_bias(name=node.name,
                                 b=second_input,
                                 input_name=node.inputs[0],
                                 output_name=node.outputs[0],
                                 shape_bias=[second_input.shape[0]])
                return

    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode="ADD"
    )

def _convert_mul(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode="MULTIPLY"
    )

def _convert_div(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_unary(name=node.name + '_inverse', #type: ignore
                      input_name=node.inputs[1],
                      output_name=node.inputs[1] + '_inverse',
                      mode='inverse')
    builder.add_elementwise(
        name=node.name,
        input_names=[node.inputs[0], node.inputs[1] + '_inverse'],
        output_name=node.outputs[0],
        mode="MULTIPLY"
    )

def _convert_leaky_relu(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    alpha = node.attrs.get('alpha', 0.01)
    builder.add_activation(
        name=node.name,
        non_linearity='LEAKYRELU',
        params=[alpha],
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_concat(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    axis = node.attrs.get("axis", 1)
    parent_op_type = graph.blob_from_op_type.get(node.inputs[0], None)
    mode = None
    first_input_shape = None

    if node.inputs[0] in graph.shape_dict:
        first_input_shape = graph.shape_dict[node.inputs[0]]
        if parent_op_type in _SEQUENCE_LAYERS_REGISTRY and \
            len(first_input_shape) == 3:
            if axis == 0:
                mode = 'SEQUENCE_CONCAT'
            if axis == 2:
                mode = 'CONCAT'
        elif (len(first_input_shape) == 1 and axis == 0) or \
            (len(first_input_shape) == 3 and axis == 0) or \
            (len(first_input_shape) == 4 and axis == 1) or \
            (len(first_input_shape) == 2 and axis == 1):
            mode = 'CONCAT'
    else: # shape info is not available. Fall back to guessing (ideally this should not happen)
        if axis == 0:
            mode = "SEQUENCE_CONCAT"
        elif axis == 1:
            mode = "CONCAT"

    if mode is None:
        return err.unsupported_op_configuration(builder, node, graph,
                                                "Unsupported axis {} in input of shape".
                                                format(axis, str(first_input_shape)))
    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode=mode
    )

def _convert_reduce(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None

    if node.op_type == 'ArgMax' or node.op_type == 'ArgMin':
        axes = [node.attrs.get('axis', 0)]
    else:
        axes = node.attrs.get('axes', None)
    if axes is None:
        assert node.inputs[0] in graph.shape_dict, "Shape inference failed for reduce op"
        shape = graph.shape_dict[node.inputs[0]]
        axes = range(0, len(shape))

    def get_coreml_axis(axes): # type: (List[int]) -> Text
        coreml_axis = ""
        assert node.inputs[0] in graph.shape_dict, "Shape inference failed for reduce op"
        input_shape = graph.shape_dict[node.inputs[0]]
        if len(input_shape) == 1: coreml_axis = 'C'
        elif len(input_shape) == 2:
            if len(axes) == 1 and axes[0] == 1: coreml_axis = 'C'
        elif len(input_shape) == 3:
            for ind in [['C','H','W'][i] for i in axes]: coreml_axis += ind
        elif len(input_shape) == 4:
            for ind in [['B','C','H','W'][i] for i in axes]: coreml_axis += ind
        return coreml_axis

    coreml_axis = get_coreml_axis(axes)

    if coreml_axis not in ['C', 'H', 'W', 'HW', 'CHW']:
        return err.unsupported_op_configuration(builder, node, graph, "Unable to translate axes attribute to CoreML axis parameter for %s" % axes)

    input_name = node.inputs[0]

    if node.op_type == 'ReduceMean':
        mode = 'avg'
    elif node.op_type == 'ReduceL1':
        mode = 'L1'
    elif node.op_type == 'ReduceL2':
        mode = 'L2'
    elif node.op_type == 'ReduceLogSum':
        return err.unsupported_op_configuration(builder, node, graph, "ReduceLogSum is not supported. Note: CoreML does support a logsum, but CoreML logsum computes sum(log(elements)), and ONNX defines logsum to be log(sum(elements))")
    elif node.op_type == 'ReduceMax':
        mode = 'max'
    elif node.op_type == 'ReduceMin':
        mode = 'min'
    elif node.op_type == 'ReduceProd':
        mode = 'prod'
    elif node.op_type == 'ReduceSum':
        mode = 'sum'
    elif node.op_type == 'ReduceSumSquare':
        mode = 'sumsquare'
    elif node.op_type == 'ArgMax':
        mode = 'argmax'
    elif node.op_type == 'ArgMin':
        mode = 'argmax'
        builder.add_elementwise(name=node.name+'_multiply_minus_1',  # type: ignore
                                input_names=[input_name],
                                output_name=input_name+'_multiply_minus_1',
                                mode='MULTIPLY',
                                alpha=-1)
        input_name += '_multiply_minus_1'
    else:
        return err.unsupported_op_configuration(builder, node, graph, "Unsupported op")

    builder.add_reduce(name=node.name,
                       input_name=input_name, output_name=node.outputs[0],
                       axis=coreml_axis,
                       mode=mode)

def _convert_softmax(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    axis = node.attrs.get('axis', 1)
    if axis != 1:
        return err.unsupported_op_configuration(builder, node, graph, "Unsupported axis {} for softmax".format(axis,))

    builder.add_softmax(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_gemm(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None

    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))

    if node.attrs["transB"] != 1:
        return err.unsupported_op_configuration(builder, node, graph, "Gemm is supported only for inner_product layer")

    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or (b is not None and len(b.shape) != 1):
        return err.unsupported_op_configuration(builder, node, graph, "Gemm is supported only for inner_product layer")

    if b is not None:
        if W.shape[0] != b.shape[0]:
            return err.unsupported_op_configuration(builder, node, graph, "Gemm is supported only for inner_product layer")

    input_channels = W.shape[1]
    output_channels = W.shape[0]
    builder.add_inner_product(
        name=node.name,
        W=W,
        b=b,
        input_channels=input_channels,
        output_channels=output_channels,
        has_bias=b is not None,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_matmul(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None

    weight_name = node.inputs[1]
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    else:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(weight_name, ))

    if len(W.shape) != 2:
        return err.unsupported_op_configuration(builder, node, graph, "Gemm is supported only for inner_product layer")

    input_channels = W.shape[0]
    output_channels = W.shape[1]
    builder.add_inner_product(
        name=node.name,
        W=np.transpose(W), # type: ignore
        b=None,
        input_channels=input_channels,
        output_channels=output_channels,
        has_bias=False,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_lrn(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    alpha = node.attrs.get("alpha", 1.0e-4)
    beta = node.attrs.get("beta", 0.75)
    bias = node.attrs.get("bias", 1.0)
    size = node.attrs["size"]
    builder.add_lrn(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        alpha=alpha,
        beta=beta,
        k=bias,
        local_size=size
    )

def _convert_sigmoid(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_activation(
        name=node.name,
        non_linearity='SIGMOID',
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_elu(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    alpha = node.attrs.get('alpha', 1.0)
    builder.add_activation(
        name=node.name,
        non_linearity='ELU',
        params=alpha,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_selu(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    alpha = node.attrs.get('alpha', 1.6732)
    gamma = node.attrs.get('gamma', 1.0507)
    builder.add_activation(
        name=node.name + '_elu', #type: ignore
        non_linearity='ELU',
        params=alpha,
        input_name=node.inputs[0],
        output_name=node.inputs[0] + '_elu'
    )
    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs[0] + '_elu',
        output_name=node.outputs[0],
        mode='MULTIPLY',
        alpha=gamma
    )

def _convert_prelu(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    slope = node.input_tensors[node.inputs[1]]
    builder.add_activation(
        name=node.name,
        non_linearity='PRELU',
        params=slope,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_tanh(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_activation(
        name=node.name,
        non_linearity='TANH',
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )

def _convert_abs(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='abs'
    )

def _convert_pad(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    mode = node.attrs['mode']
    if mode == 'reflect' or mode == b'reflect':
        mode = 'reflection'
    elif mode == 'edge' or mode == b'edge':
        mode = 'replication'
    else:
        mode = 'constant'
    pads = node.attrs['pads']
    if not (len(pads) % 2 == 0 and len(pads) >= 2):
        return err.unsupported_op_configuration(builder, node, graph,
                                         "pads attribute: {}."
                                         "Length of pads must be a multiple of 2".format(str(pads)))

    start = pads[:len(pads)//2]
    end = pads[len(pads)//2:]
    if len(start) < 2:
        start.append(0)
        end.append(0)

    def _all_zero(x):  # type: (Sequence[int]) -> bool
        return x.count(0) == len(x)

    if not _all_zero(start[:-2]) and not _all_zero(end[:-2]):
        return err.unsupported_op_configuration(builder, node, graph, "Paddings value {} not supported".format(pads,))

    pad_t = start[-2]
    pad_b = end[-2]
    pad_l = start[-1]
    pad_r = end[-1]
    value = node.attrs.get('value', 0.0)
    builder.add_padding(
        name=node.name,
        left=pad_l,
        right=pad_r,
        top=pad_t,
        bottom=pad_b,
        value=value,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        padding_type=mode
    )

def _convert_slice(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    # TODO: support multi-axis slice
    input_shape = graph.shape_dict.get(node.inputs[0], None)
    starts = node.attrs['starts']
    ends = node.attrs['ends']
    axes = node.attrs.get('axes', range(len(starts)))
    if len(axes) != 1:
        return err.unsupported_op_configuration(builder, node, graph, "Only single axis Slice is supported now")

    if input_shape and len(input_shape) == 4 and len(axes) == 1:
        axis = ['B','channel','height','width'][axes[0]]
    elif len(axes) == 1:
        if axes[0] == 0:
            axis = 'channel'
        elif axes[0] == 1:
            axis = 'height'
        elif axes[0] == 2:
            axis = 'width'
        else:
            return err.unsupported_op_configuration(builder, node, graph, "Slice is supported only along H, W or C dimensions")
    else:
        return err.unsupported_op_configuration(builder, node, graph, "Slice is supported only along one axis for 3D or 4D Tensors")

    builder.add_slice(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        axis=axis,
        start_index=starts[0],
        end_index=ends[0],
        stride=1
    )

def _convert_exp(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='exp'
    )

def _convert_flatten(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_flatten(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode=0
    )

def _convert_max(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    if len(node.inputs) == 1:
        inputs = [node.inputs[0], node.inputs[0]]
    else:
        inputs = node.inputs
    builder.add_elementwise(
        name=node.name,
        input_names=inputs,
        output_name=node.outputs[0],
        mode='MAX'
    )

def _convert_min(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    if len(node.inputs) == 1:
        inputs = [node.inputs[0], node.inputs[0]]
    else:
        inputs = node.inputs
    builder.add_elementwise(
        name=node.name,
        input_names=inputs,
        output_name=node.outputs[0],
        mode='MIN'
    )

def _convert_softsign(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_activation(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        non_linearity='SOFTSIGN'
    )

def _convert_softplus(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_activation(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        non_linearity='SOFTPLUS'
    )

def _convert_hardsigmoid(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    alpha = node.attrs.get('alpha', 0.2)
    beta = node.attrs.get('beta', 0.5)
    builder.add_activation(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        non_linearity='SIGMOID_HARD',
        params = [alpha, beta]
    )

def _convert_logsoftmax(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    axis = node.attrs.get('axis', 1)
    if axis != 1:
        return err.unsupported_op_configuration(builder, node, graph, "Unsupported axis {} for logsoftmax".format(axis,))

    builder.add_softmax(
        name=node.name + '_softmax', #type: ignore
        input_name=node.inputs[0],
        output_name=node.outputs[0] + '_softmax'
    )
    builder.add_unary(
        name=node.name,
        input_name=node.outputs[0] + '_softmax',
        output_name=node.outputs[0],
        mode='log'
    )


def _convert_neg(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode='MULTIPLY',
        alpha=-1.0
    )

def _convert_split(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    axis = node.attrs.get("axis", 0)
    if not (axis == 0 or axis == 1):
        return err.unsupported_op_configuration(builder, node, graph, "Unsupported axis {}".format(axis, ))
    builder.add_split(
        name=node.name,
        input_name=node.inputs[0],
        output_names=node.outputs
    )

def _convert_log(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='log'
    )

def _convert_sqrt(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='sqrt'
    )

def _convert_reciprocal(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='inverse'
    )

def _convert_reorganize_data(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    mode = 'SPACE_TO_DEPTH'
    if node.op_type == 'DepthToSpace':
        mode = 'DEPTH_TO_SPACE'
    block_size = node.attrs.get('blocksize', 2)
    builder.add_reorganize_data(name = node.name,
         input_name = node.inputs[0],
         output_name = node.outputs[0],
         mode=mode,
         block_size=block_size
    )

def _convert_upsample(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    scales = node.attrs["scales"]
    if len(scales) != 4 or scales[0] != 1.0 or scales[1] != 1.0:
        err.unsupported_op_configuration(builder, node, graph, "Unsupported scales {} for upsample".format(scales))
    height_scale = int(scales[2])
    width_scale = int(scales[3])
    mode_convert = {
        "nearest": "NN",
        "bilinear": "BILINEAR",
    }
    mode = mode_convert[node.attrs["mode"].decode("UTF-8")]
    builder.add_upsample(
        name=node.name,
        scaling_factor_h=height_scale,
        scaling_factor_w=width_scale,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode=mode,
    )

def _convert_clip(builder, node, graph, err): # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    max_limit = node.attrs.get('max',float(2^16-1))
    min_limit = node.attrs.get('min',float(-(2^16-1)))
    delta = max_limit - min_limit
    builder.add_activation(name = node.name + '_scale_0_1', # type: ignore
                           non_linearity = 'LINEAR',
                           input_name = node.inputs[0],
                           output_name = node.inputs[0] + '_scale_0_1',
                           params = [1.0/delta, -min_limit/delta])
    builder.add_activation(name = node.name + '_clip_0_1', # type: ignore
                           non_linearity = 'SIGMOID_HARD',
                           input_name = node.inputs[0] + '_scale_0_1',
                           output_name = node.inputs[0] + '_clip_0_1',
                           params = [1.0, 0.0])
    builder.add_activation(name = node.name,
                           non_linearity = 'LINEAR',
                           input_name = node.inputs[0] + '_clip_0_1',
                           output_name = node.outputs[0],
                           params = [delta, min_limit])


def _convert_mvn(builder, node, graph, err): # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_mvn(name=node.name,
                    input_name=node.inputs[0],
                    output_name=node.outputs[0],
                    across_channels = node.attrs.get('across_channels', 0),
                    normalize_variance = node.attrs.get('normalize_variance', 1),
                    epsilon = 1e-5)

def _convert_lstm(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    W_name = node.inputs[1]
    R_name = node.inputs[2]
    B = None
    if len(node.inputs) > 3:
        B_name = node.inputs[3]
        B = node.input_tensors.get(B_name, None)
    W = node.input_tensors.get(W_name, None)
    R = node.input_tensors.get(R_name, None)
    if W is None:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(W_name, ))
    if R is None:
        err.missing_initializer(node,
                                "Weight tensor: {} not found in the graph initializer".format(R_name, ))

    h = node.attrs["hidden_size"]
    W_i, W_o, W_f, W_c = np.split(W, 4)  #type: ignore
    R_i, R_o, R_f, R_c = np.split(R, 4)  #type: ignore
    x = W_i.shape[1]
    W_x = [W_i, W_f, W_o, W_c]
    W_h = [R_i, R_f, R_o, R_c]
    b = None
    if B is not None:
        b_Wi, b_Wo, b_Wf, b_Wc, b_Ri, b_Ro, b_Rf, b_Rc = np.split(B, 8)  #type: ignore
        b = [b_Wi + b_Ri, b_Wf + b_Rf, b_Wo + b_Ro, b_Wc + b_Rc]

    input_h = node.inputs[5] if len(node.inputs) > 5 else node.inputs[0] + '_h_input'
    input_c = node.inputs[6] if len(node.inputs) > 6 else node.inputs[0] + '_c_input'
    output_h = node.outputs[1] if len(node.outputs) > 1 else node.outputs[0] + '_h_output'
    output_c = node.outputs[2] if len(node.outputs) > 2 else node.outputs[0] + '_c_output'

    builder.add_unilstm(name = node.name,
                    W_h = W_h,
                    W_x = W_x,
                    b = b,
                    hidden_size = h,
                    input_size = x,
                    input_names= [node.inputs[0], input_h, input_c],
                    output_names= [node.outputs[0], output_h, output_c],
                    inner_activation='SIGMOID',
                    cell_state_update_activation='TANH',
                    output_activation='TANH',
                    peep=None,
                    output_all=True,
                    forget_bias=False, coupled_input_forget_gate=False,
                    cell_clip_threshold=50000.0, reverse_input=False)


def _convert_custom(builder, node, graph, err): # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None

    if node.op_type in err.custom_conversion_functions:
        func = err.custom_conversion_functions[node.op_type]
        params = func(node)
    else:
        params = NeuralNetwork_pb2.CustomLayerParams()
        params.className = node.op_type
        params.description = "Custom layer that corresponds to the ONNX op {}".format(node.op_type,)

    inputs_ = []
    # skip the inputs that are initializers
    for inp in node.inputs:
        if inp not in node.input_tensors:
            inputs_.append(inp)

    builder.add_custom(name=node.name,
                       input_names=inputs_,
                       output_names=node.outputs,
                       custom_proto_spec=params)

    err.custom_layer_nodes.append(node)

def _convert_identity(builder, node, graph, err): # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    builder.add_activation(
        name=node.name,
        non_linearity = 'LINEAR',
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        params=[1.0, 0.0]
   )

def _convert_const(builder, node, graph, err): # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None

    for name, value in node.input_tensors.items(): 
        if name not in graph.constant_layers_added:
            shape = value.shape
            coreml_shape = [1,1,1]
            if len(shape) == 3:
                coreml_shape = list(shape)
            elif len(shape) == 1:
                coreml_shape = [shape[0],1,1]
            elif len(shape) == 2:
                coreml_shape = [1, shape[0], shape[1]]
            else:
                return err.unsupported_op_configuration(builder, node, graph, "unable to translate constant array shape to CoreML shape")
            builder.add_load_constant(name=name,
                                  output_name=name,
                                  constant_value=value.flatten(),
                                  shape=coreml_shape)
            graph.constant_layers_added[name] = True


_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "ConvTranspose": _convert_conv,
    "Relu": _convert_relu,
    "Reshape": _convert_reshape,
    "Transpose": _convert_transpose,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "FC": _convert_fc,
    "BatchNormalization": _convert_bn,
    "SpatialBN": _convert_bn,
    "InstanceNormalization": _convert_instancenorm,
    "Add": _convert_add,
    "Sum": _convert_add,
    "Mul": _convert_mul,
    "LeakyRelu": _convert_leaky_relu,
    "Concat": _convert_concat,
    "GlobalAveragePool": _convert_pool,
    "GlobalMaxPool": _convert_pool,
    "Softmax": _convert_softmax,
    "Gemm": _convert_gemm,
    "MatMul": _convert_matmul,
    "LRN": _convert_lrn,
    "Sigmoid": _convert_sigmoid,
    "Abs": _convert_abs,
    "Pad": _convert_pad,
    "Slice": _convert_slice,
    "Elu": _convert_elu,
    "PRelu": _convert_prelu,
    "Tanh": _convert_tanh,
    "Exp": _convert_exp,
    "Flatten": _convert_flatten,
    "Max": _convert_max,
    "Min": _convert_min,
    "Softsign": _convert_softsign,
    "Softplus": _convert_softplus,
    "Neg": _convert_neg,
    "Split": _convert_split,
    "Log": _convert_log,
    "Div": _convert_div,
    "HardSigmoid": _convert_hardsigmoid,
    "LogSoftmax": _convert_logsoftmax,
    "Reciprocal": _convert_reciprocal,
    "Selu": _convert_selu,
    "Sqrt": _convert_sqrt,
    "ThresholdedRelu": _convert_thresholdedrelu,
    "DepthToSpace": _convert_reorganize_data,
    "SpaceToDepth": _convert_reorganize_data,
    "LSTM": _convert_lstm,
    "Upsample": _convert_upsample,
    "ReduceL1": _convert_reduce,
    "ReduceL2": _convert_reduce,
    "ReduceLogSum": _convert_reduce,
    "ReduceMax": _convert_reduce,
    "ReduceMean": _convert_reduce,
    "ReduceMin": _convert_reduce,
    "ReduceProd": _convert_reduce,
    "ReduceSum": _convert_reduce,
    "ReduceSumSquare": _convert_reduce,
    "ArgMax": _convert_reduce,
    "ArgMin": _convert_reduce,
    "Clip": _convert_clip,
    "MeanVarianceNormalization": _convert_mvn,
    "Unsqueeze": _convert_identity,
    "Squeeze": _convert_identity
}

_SEQUENCE_LAYERS_REGISTRY = set(["LSTM"])

_CONST_INPUT_ALLOWED_LAYERS = set([ "Add", "Sum", "Mul", "Concat", "Max", "Min", "Div", "Reciprocal"])

def _get_node_converter_fn(builder, node, err):  # type: (NeuralNetworkBuilder, Node, ErrorHandling) -> Callable[[NeuralNetworkBuilder, Node, Graph, ErrorHandling], None]
    """
    Get the right converter function for ONNX node op_type
    """
    op_type = node.op_type
    if op_type in _ONNX_NODE_REGISTRY:
        return _ONNX_NODE_REGISTRY[op_type]
    else:
        return err.unsupported_op(node)

def _add_const_inputs_if_required(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    if node.op_type in _CONST_INPUT_ALLOWED_LAYERS:
        if len(node.input_tensors) > 0:
            _convert_const(builder, node, graph, err)


def _convert_node(builder, node, graph, err):  # type: (NeuralNetworkBuilder, Node, Graph, ErrorHandling) -> None
    converter_fn = _get_node_converter_fn(builder, node, err)
    return converter_fn(builder, node, graph, err)
