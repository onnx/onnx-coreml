from typing import NamedTuple, Sequence, Dict, Text
from onnx import ModelProto, GraphProto, helper
try:
    import caffe2
    from caffe2.python.onnx.workspace import Workspace
    from caffe2.python.onnx.backend_rep import Caffe2Rep
    from caffe2.python.onnx import backend
    CAFFE2_AVAILABLE = True
except ImportError:
    CAFFE2_AVAILABLE = False


# Infer shapes for all intermediate values and write them into GraphProto.value_info
def infer_shapes_and_types(graph):  # type: (GraphProto) -> None
    if not CAFFE2_AVAILABLE:
        return
    model = helper.make_model(graph)
    try:
        c2_backend = _load_model(model)
        #TODO Gate the following two lines with "with c2_backend.workspace:" once fixed
        inference_result = _infer(c2_backend)
        _add_shapes_and_types_to_graph(graph, inference_result)
    except:
        return



def _load_model(onnx_model):  # type: (ModelProto) -> Caffe2Rep
    return backend.prepare(onnx_model)


_InferenceResult = NamedTuple('InferenceResult', [
    ('type', int),
    ('shape', Sequence[int]),
])


def _infer(c2_backend):  # type: (Caffe2Rep) -> Dict[Text, _InferenceResult]
    c2_backend.workspace.CreateNet(c2_backend.init_net)
    c2_backend.workspace.CreateNet(c2_backend.predict_net)
    #TODO Use c2_backend.workspace.InferShapesAndTypes once fixed
    (shapes, types) = caffe2.python.workspace.InferShapesAndTypes([
        caffe2.python.core.Net(c2_backend.init_net),
        caffe2.python.core.Net(c2_backend.predict_net),
        ])
    assert shapes.keys() == types.keys()
    shapes_and_types = {}
    for name in shapes:
        shapes_and_types[name] = _InferenceResult(
            type=types[name],
            shape=shapes[name],
        )
    return shapes_and_types


def _add_shapes_and_types_to_graph(graph, shapes_and_types):  # type: (GraphProto, Dict[Text, _InferenceResult]) -> None
    graph_inputs = {i.name: i for i in graph.input}
    graph_outputs = {o.name: o for o in graph.output}
    graph_intermediates = {i.name: i for i in graph.value_info}

    for name, shape_and_type in shapes_and_types.items():
        value_info = helper.make_tensor_value_info(name, shape_and_type.type, shape_and_type.shape)

        found = False
        if name in graph_inputs:
            graph_inputs[name] = value_info
            found = True
        if name in graph_outputs:
            graph_outputs[name] = value_info
            found = True
        if name in graph_intermediates:
            graph_intermediates[name] = value_info
            found = True
        if not found:
            graph.value_info.extend([value_info])
