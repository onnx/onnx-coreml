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
    def run_node(cls, node, inputs, device='CPU'):
        super(CoreMLBackend, cls).run_node(node, inputs, device)
        # TODO: implement single node run
        raise NotImplementedError('Not implemented')

    @classmethod
    def supports_device(cls, device):
        # supports only CPU for testing as GPU CoreML generates different
        # results as of fp16 usage
        return device == 'CPU'


prepare = CoreMLBackend.prepare

run_node = CoreMLBackend.run_node

run_model = CoreMLBackend.run_model
