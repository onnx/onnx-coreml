from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx

import onnx.backend.test
import caffe2.python.onnx.backend

from onnx_coreml._backend import CoreMLBackend


# TODO: don't use caffe2 to infer output shapes
class CoreMLTestingBackend(CoreMLBackend):
    @classmethod
    def run_node(cls, node, inputs, device='CPU'):
        '''
        CoreML requires full model for prediction, not just single layer.
        Also input/output shapes are required to build CoreML spec for model.
        As a temporary decision we use caffe2 backend for shape inference
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

        c2_result = caffe2.python.onnx.backend.run_node(node, inputs, device)

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

# unsupported tests
backend_test.exclude('test_clip')
backend_test.exclude('test_softmax_lastdim')
backend_test.exclude('test_softmax_functional_dim3')
backend_test.exclude('test_log_softmax_dim3')
backend_test.exclude('test_log_softmax_lastdim')
backend_test.exclude('test_ZeroPad2d_negative_dims')
backend_test.exclude('test_Softsign')  # test uses broadcasted add
backend_test.exclude('test_Softmax')  # arbitrary dimensional
backend_test.exclude('test_Softmin')  # arbitrary dimensional
backend_test.exclude('test_Embedding')  # Gather op

# input of unsupported shape is used. coreml currently
# supports 1d or 3d (shape (C, H, W) only)
backend_test.exclude('test_PReLU')
backend_test.exclude('test_Conv3d')
backend_test.exclude('test_BatchNorm3d')
backend_test.exclude('test_BatchNorm1d')
backend_test.exclude('test_AvgPool2d')
backend_test.exclude('test_AvgPool3d')
backend_test.exclude('test_GLU')
backend_test.exclude('test_Linear')
backend_test.exclude('test_LogSoftmax')
backend_test.exclude('test_MaxPool1d')
backend_test.exclude('test_MaxPool2d')
backend_test.exclude('test_MaxPool3d')

# Exclude big models tests
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_resnet50')
backend_test.exclude('test_vgg16')
backend_test.exclude('test_vgg19')
backend_test.exclude('test_densenet121')
backend_test.exclude('test_inception_v1')
backend_test.exclude('test_inception_v2')
backend_test.exclude('test_shufflenet')
backend_test.exclude('test_squeezenet')

globals().update(backend_test
                 .enable_report()
                 .test_cases)


if __name__ == '__main__':
    unittest.main()
