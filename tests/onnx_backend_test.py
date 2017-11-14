from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx

import onnx.backend.test
import onnx_caffe2.backend

from onnx_coreml._backend import CoreMLBackend


# TODO: don't use onnx_caffe2 to infer output shapes
class CoreMLTestingBackend(CoreMLBackend):
    @classmethod
    def run_node(cls, node, inputs, device='CPU'):
        '''
        CoreML requires full model for prediction, not just single layer.
        Also input/output shapes are required to build CoreML spec for model.
        As a temporary decision we use onnx_caffe2 backend for shape inference
        task to build the appropriate ONNX model and convert it to
        CoreML model.
        '''
        super(CoreMLTestingBackend, cls).run_node(node, inputs, device)

        graph_inputs = []
        for i in range(len(inputs)):
            input_ = inputs[i]
            value_info = onnx.helper.make_tensor_value_info(
                name=node.input[i],
                elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_.dtype],
                shape=input_.shape
            )
            graph_inputs.append(value_info)

        c2_result = onnx_caffe2.backend.run_node(node, inputs, device)

        graph_outputs = []
        for i in range(len(node.output)):
            c2_output = c2_result[i]
            value_info = onnx.helper.make_tensor_value_info(
                name=node.output[i],
                elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[c2_output.dtype],
                shape=c2_output.shape
            )
            graph_outputs.append(value_info)

        graph = onnx.helper.make_graph(
            nodes=[node],
            name='dummy',
            inputs=graph_inputs,
            outputs=graph_outputs
        )

        model = onnx.helper.make_model(graph)
        return cls.prepare(model).run(inputs)


# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(CoreMLTestingBackend, __name__)
backend_test.exclude('test_constant')
backend_test.exclude('test_matmul')
backend_test.exclude('test_slice')
backend_test.exclude('test_default_axes')
backend_test.exclude('test_add_bcast')


globals().update(backend_test
                 .enable_report()
                 .test_cases)


if __name__ == '__main__':
    unittest.main()
