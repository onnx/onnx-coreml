from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx.backend.base import Backend
from onnx_coreml._backend_rep import CoreMLRep
from onnx_coreml import convert


class CoreMLBackend(Backend):
    @classmethod
    def prepare(cls, model, device='CPU', **kwargs):
        super(CoreMLBackend, cls).prepare(model, device, **kwargs)
        coreml_model = convert(model)
        return CoreMLRep(coreml_model, device == 'CPU')

    @classmethod
    def supports_device(cls, device):
        return device == 'CPU'


prepare = CoreMLBackend.prepare

run_node = CoreMLBackend.run_node

run_model = CoreMLBackend.run_model
