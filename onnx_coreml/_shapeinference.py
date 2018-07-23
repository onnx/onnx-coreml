from typing import NamedTuple, Sequence, Dict, Text
from onnx import ModelProto, GraphProto, helper, shape_inference


# Infer shapes for all intermediate values and write them into GraphProto.value_info
def infer_shapes_and_types(graph):  # type: (GraphProto) -> GraphProto
    model = helper.make_model(graph)
    model_with_shapes = shape_inference.infer_shapes(model)
    return model_with_shapes.graph
