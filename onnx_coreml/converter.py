from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Text, Union, Optional, Dict, Any, Iterable, Sequence, Callable, List
from ._shapeinference import infer_shapes_and_types

import onnx
import numpy as np

from onnx import TensorProto

from coremltools.models.neural_network import NeuralNetworkBuilder  #type: ignore
from coremltools.models import datatypes, MLModel  #type: ignore
from coremltools.proto import FeatureTypes_pb2 as ft  #type: ignore

from typing import Tuple

from ._operators import _convert_node, _SEQUENCE_LAYERS_REGISTRY, _ONNX_NODE_REGISTRY, _add_const_inputs_if_required
from ._graph import Graph, EdgeInfo, Transformer
from ._transformers import ConvAddFuser, DropoutRemover, \
    ReshapeInitTensorFuser, BNBroadcastedMulFuser, BNBroadcastedAddFuser, \
    PixelShuffleFuser, OutputRenamer, AddModelInputsOutputs, \
    ConstantsToInitializers, ImageScalerRemover, UnsqueezeConstantRemover, TransposeConstantRemover, \
    ShapeOpRemover, SliceConstantRemover, ConcatConstantRemover
from ._error_utils import ErrorHandling

'''
inputs: list of tuples.
      [Tuple]: [(name, type, shape)]
'''
def _make_coreml_input_features(graph): # type: (...) -> Sequence[Tuple[Text, datatypes.Array]]
    '''
    ONNX shapes to CoreML static shapes mapping
    length==1: [C]
    length==2: [B,C]
    length==3: [C,H,W] or [Seq,B,C]
    length==4: [B,C,H,W]
    '''
    inputs = graph.inputs
    op_types = graph.blob_to_op_type
    features = []
    for input_ in inputs:
        shape = input_[2]
        if len(shape) == 0:
            shape = [1, 1, 1]
        elif len(shape) == 1:
            # assume [C]
            pass
        elif len(shape) == 2:
            # assume [Batch,C]
            shape = [shape[1]]
        elif len(shape) == 3:
            # assume [C,H,W] unless its connected to recurrent related ops
            if input_[0] in op_types and \
                len(op_types[input_[0]]) == 1 and \
                str(op_types[input_[0]][0]) in _SEQUENCE_LAYERS_REGISTRY:
                # onnx shape: (Seq,B,C)
                shape = [shape[2]]
        elif len(shape) == 4:  # (B,C,H,W) --> (C,H,W)
            shape = shape[1:]
        else:
            raise ValueError("Unrecognized input shape %s, for input '%s' " % (str(shape), str(input_[0])))
        features.append((str(input_[0]), datatypes.Array(*shape)))
    return features

'''
outputs: list of tuples.
      [Tuple]: [(name, type, shape)]
'''
def _make_coreml_output_features(graph):  # type: (...) -> Sequence[Tuple[Text, datatypes.Array]]
    features = []
    outputs = graph.outputs
    op_types = graph.blob_from_op_type
    for output_ in outputs:
        shape = output_[2]
        if len(shape) == 0:
            shape = [1, 1, 1]
        elif len(shape) == 1:
            pass
        elif len(shape) == 3:
            if output_[0] in op_types and \
                str(op_types[output_[0]]) in _SEQUENCE_LAYERS_REGISTRY:
                # onnx shape: (Seq,B,C)
                shape = [shape[2]]
        elif len(shape) == 4:  # (B,C,H,W) --> (C,H,W)
            shape = shape[1:]
        else:
            shape = None #output shape need not be specified for CoreML.
        if shape is None:
            features.append((str(output_[0]), shape))
        else:
            features.append((str(output_[0]), datatypes.Array(*shape)))
    return features

def _check_unsupported_ops(nodes): # type: (...) -> None
    unsupported_op_types = [] # type: List[Text]
    for node in nodes:
        if node.op_type not in _ONNX_NODE_REGISTRY and \
          node.op_type not in unsupported_op_types:
            unsupported_op_types.append(node.op_type)

    if len(unsupported_op_types) > 0:
        raise NotImplementedError("Unsupported ONNX ops of type: %s" % (
            ','.join(unsupported_op_types)))


def _update_multiarray_to_float32(feature, #type: Any
                                 ): # type : (...) -> None
  if feature.type.HasField('multiArrayType'):
    feature.type.multiArrayType.dataType = ft.ArrayFeatureType.FLOAT32

def _update_multiarray_to_int32(feature, #type: Any
                               ): # type : (...) -> None
  if feature.type.HasField('multiArrayType'):
    feature.type.multiArrayType.dataType = ft.ArrayFeatureType.INT32


def _transform_coreml_dtypes(builder, # type : NeuralNetworkBuilder
                             inputs, # type: List[EdgeInfo]
                             outputs # type: List[EdgeInfo]
                             ):
    # type: (...) -> None

    ''' Make sure ONNX input/output data types are mapped to the equivalent CoreML types
    '''
    for i, input_ in enumerate(inputs):
        onnx_type = input_[1]
        if onnx_type == TensorProto.FLOAT:
            _update_multiarray_to_float32(builder.spec.description.input[i])
        elif onnx_type == TensorProto.DOUBLE:
            continue
        elif onnx_type == TensorProto.INT32 or onnx_type == TensorProto.INT64:
            _update_multiarray_to_int32(builder.spec.description.input[i])
        else:
            raise TypeError("Input must be of of type FLOAT, DOUBLE, INT32 or INT64")

    for i, output_ in enumerate(outputs):
        onnx_type = output_[1]
        if onnx_type == TensorProto.FLOAT:
            _update_multiarray_to_float32(builder.spec.description.output[i])
        elif onnx_type == TensorProto.DOUBLE:
            continue
        elif onnx_type == TensorProto.INT32 or onnx_type == TensorProto.INT64:
            _update_multiarray_to_int32(builder.spec.description.output[i])
        else:
            raise TypeError("Output must be of of type FLOAT, DOUBLE, INT32 or INT64")

def _convert_multiarray_output_to_image(spec,  # type: Any
                                        feature_name,  # type: Text
                                        is_bgr=False,  # type: bool
                                        ):
    # type: (...) -> None
    for output in spec.description.output:
        if output.name != feature_name:
            continue
        if output.type.WhichOneof('Type') != 'multiArrayType':
            raise ValueError(
                "{} is not a multiarray type".format(output.name,)
            )
        array_shape = tuple(output.type.multiArrayType.shape)
        if len(array_shape) == 2:
            height, width = array_shape
            output.type.imageType.colorSpace = \
                ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
        else:
            if len(array_shape) == 4:
                if array_shape[0] != 1:
                    raise ValueError(
                        "Shape {} is not supported for image output"
                        .format(array_shape,)
                    )
                array_shape = array_shape[1:]

            channels, height, width = array_shape

            if channels == 1:
                output.type.imageType.colorSpace = \
                    ft.ImageFeatureType.ColorSpace.Value('GRAYSCALE')
            elif channels == 3:
                if is_bgr:
                    output.type.imageType.colorSpace = \
                        ft.ImageFeatureType.ColorSpace.Value('BGR')
                else:
                    output.type.imageType.colorSpace = \
                        ft.ImageFeatureType.ColorSpace.Value('RGB')
            else:
                raise ValueError(
                    "Channel Value {} is not supported for image output"
                    .format(channels,)
                )

        output.type.imageType.width = width
        output.type.imageType.height = height


def _set_deprocessing(is_grayscale,  # type: bool
                      builder,  # type: NeuralNetworkBuilder
                      deprocessing_args,  # type: Dict[Text, Any]
                      input_name,  # type: Text
                      output_name,  # type: Text
                      ):
    # type: (...) -> None
    is_bgr = deprocessing_args.get('is_bgr', False)

    image_scale = deprocessing_args.get('image_scale', 1.0)

    if is_grayscale:
        gray_bias = deprocessing_args.get('gray_bias', 0.0)
        W = np.array([image_scale])
        b = np.array([gray_bias])
    else:
        W = np.array([image_scale, image_scale, image_scale])

        red_bias = deprocessing_args.get('red_bias', 0.0)
        green_bias = deprocessing_args.get('green_bias', 0.0)
        blue_bias = deprocessing_args.get('blue_bias', 0.0)

        if not is_bgr:
            b = np.array([
                red_bias,
                green_bias,
                blue_bias,
            ])
        else:
            b = np.array([
                blue_bias,
                green_bias,
                red_bias,
            ])
    builder.add_scale(
        name=input_name,
        W=W,
        b=b,
        has_bias=True,
        shape_scale=W.shape,
        shape_bias=b.shape,
        input_name=input_name,
        output_name=output_name
    )


def _prepare_onnx_graph(graph, transformers):  # type: (Graph, Iterable[Transformer]) -> Graph
    graph = infer_shapes_and_types(graph)
    graph_ = Graph.from_onnx(graph)
    #from .graph_viz import plot_graph
    #plot_graph(graph_, '/tmp/graph_raw.png')
    graph_ = graph_.transformed(transformers)
    #plot_graph(graph_, '/tmp/graph_opt.png')
    return graph_

def convert(model,  # type: Union[onnx.ModelProto, Text]
            mode=None,  # type: Optional[Text]
            image_input_names=[],  # type: Sequence[Text]
            preprocessing_args={},  # type: Dict[Text, Any]
            image_output_names=[],  # type: Sequence[Text]
            deprocessing_args={},  # type: Dict[Text, Any]
            class_labels=None,  # type: Union[Text, Iterable[Text], None]
            predicted_feature_name='classLabel',  # type: Text
            add_custom_layers = False,  # type: bool
            custom_conversion_functions = {}, #type: Dict[Text, Any]
            ):
    # type: (...) -> MLModel
    """
    Convert ONNX model to CoreML.
    Parameters
    ----------
    model:
        An ONNX model with parameters loaded in onnx package or path to file
        with models.
    mode: 'classifier', 'regressor' or None
        Mode of the converted coreml model:
        'classifier', a NeuralNetworkClassifier spec will be constructed.
        'regressor', a NeuralNetworkRegressor spec will be constructed.
    preprocessing_args:
        'is_bgr', 'red_bias', 'green_bias', 'blue_bias', 'gray_bias',
        'image_scale' keys with the same meaning as
        https://apple.github.io/coremltools/generated/coremltools.models.neural_network.html#coremltools.models.neural_network.NeuralNetworkBuilder.set_pre_processing_parameters
    deprocessing_args:
        Same as 'preprocessing_args' but for deprocessing.
    class_labels:
        As a string it represents the name of the file which contains
        the classification labels (one per line).
        As a list of strings it represents a list of categories that map
        the index of the output of a neural network to labels in a classifier.
    predicted_feature_name:
        Name of the output feature for the class labels exposed in the Core ML
        model (applies to classifiers only). Defaults to 'classLabel'
    add_custom_layers: bool
        Flag to turn on addition of custom CoreML layers for unsupported ONNX ops or attributes within
        a supported op.
    custom_conversion_functions: dict()
        A dictionary with keys corresponding to the names of onnx ops and values as functions taking
        an object of class 'Node' (see onnx-coreml/_graph.Node) and returning CoreML custom layer parameters.
    Returns
    -------
    model: A coreml model.
    """
    if isinstance(model, Text):
        onnx_model = onnx.load(model)
    elif isinstance(model, onnx.ModelProto):
        onnx_model = model
    else:
        raise TypeError(
            "Model must be file path to .onnx file or onnx loaded model"
        )

    transformers = [
        ConstantsToInitializers(),
        ShapeOpRemover(),
        ReshapeInitTensorFuser(),
        DropoutRemover(),
        UnsqueezeConstantRemover(),
        TransposeConstantRemover(),
        SliceConstantRemover(),
        ConcatConstantRemover(),
        ConvAddFuser(),
        BNBroadcastedMulFuser(),
        BNBroadcastedAddFuser(),
        PixelShuffleFuser(),
        AddModelInputsOutputs(),
    ]  # type: Iterable[Transformer]

    graph = _prepare_onnx_graph(onnx_model.graph, transformers)

    # are there ImageScaler nodes in the Graph?
    # If yes then add the info from it to the preprocessing dictionary, if the dictionary is not
    # already provided by the user
    if not bool(preprocessing_args):
        for node in graph.nodes:
            if node.op_type == 'ImageScaler':
                inp_name = node.inputs[0]
                scale = node.attrs.get('scale', 1.0)
                bias = node.attrs.get('bias', [0,0,0])
                if not (len(bias) == 1 or len(bias) == 3):
                    continue
                if 'image_scale' in preprocessing_args:
                    preprocessing_args['image_scale'][inp_name] = scale
                else:
                    preprocessing_args['image_scale'] = {inp_name: scale}
                if len(bias) == 3:
                    for i, color in enumerate(['red', 'green', 'blue']):
                        if color + '_bias' in preprocessing_args:
                            preprocessing_args[color + '_bias'][inp_name] = bias[i]
                        else:
                            preprocessing_args[color + '_bias'] = {inp_name: bias[i]}
                else:
                    if 'gray_bias' in preprocessing_args:
                        preprocessing_args['gray_bias'][inp_name] = bias[0]
                    else:
                        preprocessing_args['gray_bias'] = {inp_name: bias[0]}
                if inp_name not in image_input_names:
                    image_input_names.append(inp_name) # type: ignore

    # remove all ImageScaler ops
    graph = graph.transformed([ImageScalerRemover()])

    #Make CoreML input and output features by gathering shape info and
    #interpreting it for CoreML
    input_features = _make_coreml_input_features(graph)
    output_features = _make_coreml_output_features(graph)

    builder = NeuralNetworkBuilder(input_features, output_features, mode = mode)
    _transform_coreml_dtypes(builder, graph.inputs, graph.outputs)

    is_deprocess_bgr_only = (len(deprocessing_args) == 1) and \
                            ("is_bgr" in deprocessing_args)
    add_deprocess = (len(image_output_names) > 0) and \
                    (len(deprocessing_args) > 0) and \
                    (not is_deprocess_bgr_only)

    if add_deprocess:
        mapping = {}
        for f in output_features:
            output_name = f[0]
            mapping[output_name] = graph.get_unique_edge_name(output_name)
        graph = OutputRenamer(mapping)(graph)



    if len(image_input_names) > 0:
        builder.set_pre_processing_parameters(
            image_input_names=image_input_names,
            is_bgr=preprocessing_args.get('is_bgr', False),
            red_bias=preprocessing_args.get('red_bias', 0.0),
            green_bias=preprocessing_args.get('green_bias', 0.0),
            blue_bias=preprocessing_args.get('blue_bias', 0.0),
            gray_bias=preprocessing_args.get('gray_bias', 0.0),
            image_scale=preprocessing_args.get('image_scale', 1.0)
        )

    preprocessing_args.clear()

    if len(image_output_names) > 0:
        for f in output_features:
            f_name = f[0]
            if f_name in image_output_names:
                is_bgr = deprocessing_args.get('is_bgr', False)
                _convert_multiarray_output_to_image(
                    builder.spec, f_name, is_bgr=is_bgr
                )

    '''Iterate through all the ops and translate them to CoreML layers.
    '''
    if not add_custom_layers:
        _check_unsupported_ops(graph.nodes)

    err = ErrorHandling(add_custom_layers,
                        custom_conversion_functions)

    for i, node in enumerate(graph.nodes):
        print("%d/%d: Converting Node Type %s" %(i+1, len(graph.nodes), node.op_type))
        _add_const_inputs_if_required(builder, node, graph, err)
        _convert_node(builder, node, graph, err)

    if add_deprocess:
        for f in output_features:
            output_name = f[0]
            if output_name not in image_output_names:
                continue
            output_shape = f[1].dimensions
            if len(output_shape) == 2 or output_shape[0] == 1:
                is_grayscale = True
            elif output_shape[0] == 3:
                is_grayscale = False
            else:
                raise ValueError('Output must be RGB image or Grayscale')
            _set_deprocessing(
                is_grayscale,
                builder,
                deprocessing_args,
                mapping[output_name],
                output_name
            )

    if class_labels is not None:
        if isinstance(class_labels, Text):
            labels = [l.strip() for l in open(class_labels).readlines()]  # type: Sequence[Text]
        elif isinstance(class_labels, list):
            labels = class_labels
        else:
            raise TypeError(
                "synset variable of unknown type. Type found: {}. \
                Expected either string or list of strings."
                .format(type(class_labels),))

        builder.set_class_labels(
            class_labels=labels,
            predicted_feature_name=predicted_feature_name
        )

    # add description to inputs/outputs that feed in/out of recurrent layers
    for node_ in graph.nodes:
        if str(node_.op_type) in _SEQUENCE_LAYERS_REGISTRY:
            input_ = node_.inputs[0]
            output_ = node_.outputs[0]
            for i, inputs in enumerate(builder.spec.description.input):
                if inputs.name == input_:
                    builder.spec.description.input[i].shortDescription = 'This input is a sequence'
            for i, outputs in enumerate(builder.spec.description.output):
                if outputs.name == output_:
                    builder.spec.description.output[i].shortDescription = 'This output is a sequence'

    print("Translation to CoreML spec completed. Now compiling the CoreML model.")
    try:
        mlmodel = MLModel(builder.spec)
    except:
        raise ValueError('Compilation failed. Translation to CoreML spec was incorrect.')


    # print information about all ops for which custom layers have been added
    if len(err.custom_layer_nodes) > 0:
        print('\n')
        print("Custom layers have been added to the CoreML model "
              "corresponding to the following ops in the onnx model: ")
        for i, node in enumerate(err.custom_layer_nodes):
            input_info = []
            for input_ in node.inputs:
                input_info.append((str(input_), graph.shape_dict.get(input_, str("Shape not available"))))
            output_info = []
            for output_ in node.outputs:
                output_info.append((str(output_), graph.shape_dict.get(output_, str("Shape not available"))))
            print("{}/{}: op type: {}, op input names and shapes: {}, op output names and shapes: {}".
                  format(i+1, len(err.custom_layer_nodes), node.op_type, str(input_info), str(output_info)))

    return mlmodel
