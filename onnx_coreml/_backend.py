from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx.backend.base import Backend
from onnx_coreml._backend_rep import CoreMLRep
from onnx_coreml import convert
import onnx
from ._graph import _input_from_onnx_input


def _get_onnx_outputs(model):
    """
    Takes in an onnx model and returns a dictionary 
    of onnx output names mapped to shape tuples
    """
    if isinstance(model, str):
        onnx_model = onnx.load(model)
    elif isinstance(model, onnx.ModelProto):
        onnx_model = model

    graph = onnx_model.graph
    onnx_output_dict = {}
    for o in graph.output:
        out = _input_from_onnx_input(o)
        onnx_output_dict[out[0]] = out[2]
    return onnx_output_dict


class CoreMLBackend(Backend):
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        super(CoreMLBackend, cls).prepare(model, device, **kwargs)
        with open('/tmp/node_model.onnx', 'wb') as f:
            s = model.SerializeToString()
            f.write(s)
        coreml_model = convert(model)
        coreml_model.save('/tmp/node_model.mlmodel')
        onnx_outputs = _get_onnx_outputs(model)
        return CoreMLRep(coreml_model, onnx_outputs, device == 'CPU')

    @classmethod
    def supports_device(cls, device):
        return device == 'CPU'


prepare = CoreMLBackend.prepare

run_node = CoreMLBackend.run_node

run_model = CoreMLBackend.run_model
