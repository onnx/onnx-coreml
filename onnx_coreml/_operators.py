from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


def _convert_conv(builder, node):
    #get weights for convolution
    weight_name = node.inputs[1]
    W = None
    if weight_name in node.input_tensors:
        W = node.input_tensors[weight_name]
    if W is None:
        raise ValueError(
            "For Convoltuion layer, with input name = '%s', "
            "output name = '%s' and weight name = '%s', Weight tensor not found in graph initializer"
            %(node.inputs[0], node.outputs[0], weight_name)
        )

    W = W.transpose((2, 3, 1, 0))
    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]

    dilations = node.attrs.get("dilations", [1, 1])
    groups = 1
    groups = node.attrs.get("group", 1)
    kernel_shape = node.attrs["kernel_shape"]

    pads = node.attrs.get("pads", [0, 0, 0, 0])
    strides = node.attrs["strides"]

    builder.add_convolution(
        name=node.name,
        kernel_channels=W.shape[2],
        output_channels=W.shape[3],
        height=kernel_shape[0],
        width=kernel_shape[1],
        stride_height=strides[0],
        stride_width=strides[1],
        border_mode='valid',
        groups=groups,
        W=W,
        b=b,
        has_bias=b is not None,
        is_deconv=False,
        output_shape=None,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        dilation_factors=dilations,
        padding_top=pads[0],
        padding_bottom=pads[2],
        padding_left=pads[1],
        padding_right=pads[3]
    )


def _convert_relu(builder, node):
    builder.add_activation(
        name=node.name,
        non_linearity='RELU',
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_reshape(builder, node):
    shape = tuple(node.attrs["shape"])

    def get_coreml_target_shape(target_shape):
        if len(target_shape) == 1:  # (D,)
            coreml_shape = (1, target_shape[0], 1, 1)
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
                raise ValueError("Supports only 3d and 4d tensors")
        else:
            coreml_shape = None
        return coreml_shape

    new_shape = get_coreml_target_shape(shape)

    if new_shape is None:
        raise ValueError("Unsupported shape for reshape")

    builder.add_reshape(
        name=node.name,
        target_shape=new_shape,
        mode=0,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_transpose(builder, node):
    perm = node.attrs.get("perm", [0, 3, 2, 1])
    if len(perm) > 4:
        diff = len(perm) - 4
        if all([perm[i] == i for i in range(diff)]):
            perm = [p - diff for p in perm[diff:]]
        else:
            raise ValueError("Supports only 4d tensors")
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


def _convert_pool(builder, node):
    is_global = False
    if node.op_type.startswith('Global'):
        is_global = True

    if "dilations" in node.attrs:
        dilations = node.attrs["dilations"]
        if not all([d == 1 for d in dilations]):
            raise ValueError(
                "Only [1, 1] dilations are supported now"
            )

    if node.op_type.endswith("MaxPool"):
        layer_type = "MAX"
    elif node.op_type.endswith("AveragePool"):
        layer_type = "AVERAGE"
    else:
        raise ValueError(
            "Unsupported pool type {}".format(node.op_type,)
        )

    pad_t, pad_l, pad_b, pad_r = 0, 0, 0, 0

    if is_global:
        height, width = 0, 0
        stride_height, stride_width = 1, 1
    else:
        kernel_shape = node.attrs["kernel_shape"]
        height = kernel_shape[0]
        width = kernel_shape[1]

        if "pads" in node.attrs:
            pads = node.attrs["pads"]
            pad_t = pads[0]
            pad_l = pads[1]
            pad_b = pads[2]
            pad_r = pads[3]

        strides = node.attrs["strides"]
        stride_height = strides[0]
        stride_width = strides[1]

    builder.add_pooling(
        name=node.name,
        height=height,
        width=width,
        stride_height=stride_height,
        stride_width=stride_width,
        layer_type=layer_type,
        padding_type='VALID',
        exclude_pad_area=True,
        is_global=is_global,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        padding_top=pad_t,
        padding_bottom=pad_b,
        padding_left=pad_l,
        padding_right=pad_r
    )


def _convert_fc(builder, node):
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


def _convert_bn(builder, node):
    if node.attrs["is_test"] == 0:
        raise ValueError(
            "BatchNormalization supports only test mode"
        )

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


def _convert_add(builder, node):
    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            raise ValueError('Broadcast Add is not supported now')
    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode="ADD"
    )


def _convert_mul(builder, node):
    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            raise ValueError('Broadcast Multiply is not supported now')

    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode="MULTIPLY"
    )

def _convert_div(builder, node):
    if 'broadcast' in node.attrs:
        if node.attrs['broadcast'] == 1:
            raise ValueError('Broadcast Div is not supported now')

    builder.add_unary(name=node.name + '_inverse',
                      input_name=node.inputs[1],
                      output_name=node.inputs[1] + '_inverse',
                      mode='inverse')
    builder.add_elementwise(
        name=node.name,
        input_names=[node.inputs[0], node.inputs[1] + '_inverse'],
        output_name=node.outputs[0],
        mode="MULTIPLY"
    )

def _convert_leaky_relu(builder, node):
    alpha = node.attrs['alpha']
    builder.add_activation(
        name=node.name,
        non_linearity='LEAKYRELU',
        params=[alpha],
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_concat(builder, node):
    axis = node.attrs.get("axis", 1)
    if axis != 0 and axis != 1:
        raise ValueError(
            "Unsupported axis {}. Only sequence or "
            "channel axis is supported now"
            .format(axis,)
        )
    if axis == 0:
        mode = "SEQUENCE_CONCAT"
    elif axis == 1:
        mode = "CONCAT"

    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode=mode
    )


def _convert_softmax(builder, node):
    if "axis" in node.attrs:
        axis = node.attrs["axis"]
        if axis != 1:
            raise ValueError(
                "Unsupported axis {} for softmax".format(axis,)
            )
    builder.add_softmax(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_gemm(builder, node):
    if node.attrs["broadcast"] != 1 or node.attrs["transB"] != 1:
        raise ValueError(
            "Gemm is supported only for inner_product layer"
        )

    W = node.input_tensors[node.inputs[1]]
    b = None
    if len(node.inputs) > 2:
        b = node.input_tensors[node.inputs[2]]
    if len(W.shape) != 2 or len(b.shape) != 1:
        raise ValueError(
            "Gemm is supported only for inner_product layer"
        )

    if b is not None:
        if W.shape[0] != b.shape[0]:
            raise ValueError(
                "Gemm is supported only for inner_product layer"
            )

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


def _convert_lrn(builder, node):
    alpha = node.attrs["alpha"]
    beta = node.attrs["beta"]
    bias = node.attrs["bias"]
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


def _convert_sigmoid(builder, node):
    builder.add_activation(
        name=node.name,
        non_linearity='SIGMOID',
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_elu(builder, node):
    alpha = node.attrs["alpha"]
    builder.add_activation(
        name=node.name,
        non_linearity='ELU',
        params=alpha,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_prelu(builder, node):
    slope = node.input_tensors[node.inputs[1]]
    builder.add_activation(
        name=node.name,
        non_linearity='PRELU',
        params=slope,
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_tanh(builder, node):
    builder.add_activation(
        name=node.name,
        non_linearity='TANH',
        input_name=node.inputs[0],
        output_name=node.outputs[0]
    )


def _convert_abs(builder, node):
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='abs'
    )


def _convert_pad(builder, node):
    mode = node.attrs['mode']
    if mode == 'reflect':
        mode = 'reflection'
    elif mode == 'edge':
        mode = 'replication'
    else:
        mode = 'constant'
    pads = node.attrs['pads']
    assert len(pads) % 2 == 0 and len(pads) >= 2
    start = pads[:len(pads)//2]
    end = pads[len(pads)//2:]
    if len(start) < 2:
        start.append(0)
        end.append(0)

    def _all_zero(x):
        return x.count(0) == len(x)

    if not _all_zero(start[:-2]) and not _all_zero(end[:-2]):
        raise NotImplementedError(
            "Paddings value {} not supported".format(pads,)
        )
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


def _convert_slice(builder, node):
    starts = node.attrs['starts']
    ends = node.attrs['ends']
    axes = node.attrs.get('axes', [])
    if len(axes) == 0: axes = range(len(starts))
    if len(axes) == 1:
        if axes[0] == 0:
            axis = 'channel'
        elif axes[0] == 1:
            axis = 'height'
        elif axes[0] == 2:
            axis = 'width'
        raise NotImplementedError("Slice is supported only along H, W or C dimensions")
    else:
        raise NotImplementedError("Slice is supported only along one axis for 3D or 4D Tensors")
    builder.add_slice(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        axis=axis,
        start_index=starts[0],
        end_index=ends[0],
        stride=1
    )


def _convert_exp(builder, node):
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='exp'
    )


def _convert_flatten(builder, node):
    builder.add_flatten(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode=0
    )


def _convert_max(builder, node):
    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode='MAX'
    )


def _convert_softsign(builder, node):
    builder.add_activation(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        non_linearity='SOFTSIGN'
    )


def _convert_softplus(builder, node):
    builder.add_activation(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        non_linearity='SOFTPLUS'
    )


def _convert_neg(builder, node):
    builder.add_elementwise(
        name=node.name,
        input_names=node.inputs,
        output_name=node.outputs[0],
        mode='MULTIPLY',
        alpha=-1.0
    )


def _convert_split(builder, node):
    axis = node.attrs["axis"]
    if axis != 1:
        raise NotImplementedError("Split is supported for axis = 1 only")
    builder.add_split(
        name=node.name,
        input_name=node.inputs[0],
        output_names=node.outputs
    )


def _convert_log(builder, node):
    builder.add_unary(
        name=node.name,
        input_name=node.inputs[0],
        output_name=node.outputs[0],
        mode='log'
    )


_ONNX_NODE_REGISTRY = {
    "Conv": _convert_conv,
    "Relu": _convert_relu,
    "Reshape": _convert_reshape,
    "Transpose": _convert_transpose,
    "MaxPool": _convert_pool,
    "AveragePool": _convert_pool,
    "FC": _convert_fc,
    "BatchNormalization": _convert_bn,
    "SpatialBN": _convert_bn,
    "Add": _convert_add,
    "Sum": _convert_add,
    "Mul": _convert_mul,
    "LeakyRelu": _convert_leaky_relu,
    "Concat": _convert_concat,
    "GlobalAveragePool": _convert_pool,
    "GlobalMaxPool": _convert_pool,
    "Softmax": _convert_softmax,
    "Gemm": _convert_gemm,
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
    "Softsign": _convert_softsign,
    "Softplus": _convert_softplus,
    "Neg": _convert_neg,
    "Split": _convert_split,
    "Log": _convert_log,
    "Div": _convert_div,
}


def _get_node_converter_fn(node):
    """
    Get the right converter function for ONNX node op_type
    """
    op_type = node.op_type
    if op_type in _ONNX_NODE_REGISTRY:
        return _ONNX_NODE_REGISTRY[op_type]
    else:
        raise TypeError(
            "ONNX node of type {} is not supported.".format(op_type,)
        )


def _convert_node(builder, node):
    converter_fn = _get_node_converter_fn(node)
    return converter_fn(builder, node)
