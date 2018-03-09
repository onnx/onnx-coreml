from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from typing import Text, Union, Optional, Dict, Any, Iterable, Sequence, Callable
from ._shapeinference import infer_shapes_and_types

import onnx
import numpy as np

from onnx import TensorProto

from coremltools.models.neural_network import NeuralNetworkBuilder  #type: ignore
from coremltools.models import datatypes, MLModel  #type: ignore
from coremltools.proto import FeatureTypes_pb2 as ft  #type: ignore

from typing import Tuple

from ._operators import _convert_node
from ._graph import Graph, EdgeInfo, Transformer
from ._transformers import ConvAddFuser, DropoutRemover, \
    DanglingOutputsRemover, ReshapeInitTensorFuser, \
    BNBroadcastedMulFuser, BNBroadcastedAddFuser, PixelShuffleFuser, \
    OutputRenamer

'''
inputs: list of tuples. 
      [Tuple]: [(name, type, shape)]
'''
def _make_coreml_input_features(inputs): # type: (...) -> Sequence[Tuple[Text, datatypes.Array]]
    features = []
    for input_ in inputs:
        if input_[1] != TensorProto.FLOAT:
            raise TypeError("Input must be of of type TensorProto.FLOAT")
        shape = input_[2]
        if len(shape) == 0:
            shape = [1, 1, 1]
        elif len(shape) == 1:
            pass
        elif len(shape) == 2:
            # assume [H,W], so map to [1,H,W]
            shape = [1,shape[0],shape[1]]
        elif len(shape) == 3:
            pass #[C,H,W]
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
def _make_coreml_output_features(outputs):  # type: (...) -> Sequence[Tuple[Text, datatypes.Array]]
    features = []
    for output_ in outputs:
        if output_[1] != TensorProto.FLOAT:
            raise TypeError("Output must be of of type TensorProto.FLOAT")
        shape = output_[2]
        if len(shape) == 0:
            shape = [1, 1, 1]
        elif len(shape) == 1:
            pass
        elif len(shape) == 3:
            pass
        else:
            shape = None #output shape need not be specified for CoreML.
        if shape is None:
            features.append((str(output_[0]), shape))
        else:
            features.append((str(output_[0]), datatypes.Array(*shape)))
    return features

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
    infer_shapes_and_types(graph)
    graph_ = Graph.from_onnx(graph)
    return graph_.transformed(transformers)


def convert(model,  # type: Union[onnx.ModelProto, Text]
            mode=None,  # type: Optional[Text]
            image_input_names=[],  # type: Sequence[Text]
            preprocessing_args={},  # type: Dict[Text, Any]
            image_output_names=[],  # type: Sequence[Text]
            deprocessing_args={},  # type: Dict[Text, Any]
            class_labels=None,  # type: Union[Text, Iterable[Text], None]
            predicted_feature_name='classLabel',  # type: Text
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
        ReshapeInitTensorFuser(),
        DropoutRemover(),
        ConvAddFuser(),
        BNBroadcastedMulFuser(),
        BNBroadcastedAddFuser(),
        PixelShuffleFuser(),
        DanglingOutputsRemover()
    ]  # type: Iterable[Transformer]

    graph = _prepare_onnx_graph(onnx_model.graph, transformers)

    #Make CoreML input and output features by gathering shape info and
    #interpreting it for CoreML
    input_features = _make_coreml_input_features(graph.inputs)
    output_features = _make_coreml_output_features(graph.outputs)

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

    builder = NeuralNetworkBuilder(input_features, output_features, mode)

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

    if len(image_output_names) > 0:
        for f in output_features:
            f_name = f[0]
            if f_name in image_output_names:
                is_bgr = deprocessing_args.get('is_bgr', False)
                _convert_multiarray_output_to_image(
                    builder.spec, f_name, is_bgr=is_bgr
                )

    for i, node in enumerate(graph.nodes):
        print("%d/%d: Converting Node Type %s" %(i+1, len(graph.nodes), node.op_type))
        _convert_node(builder, node)

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

    return MLModel(builder.spec)
