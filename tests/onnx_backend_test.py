from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx
from typing import Any, Text
import onnx.backend.test
import caffe2.python.onnx.backend

from onnx_coreml._backend import CoreMLBackend


# TODO: don't use caffe2 to infer output shapes
class CoreMLTestingBackend(CoreMLBackend):
    @classmethod
    def run_node(cls,
                 node,  # type: onnx.NodeProto
                 inputs,  # type: Any
                 device='CPU',  # type: Text
                 ):
        # type: (...) -> onnx.namedtupledict
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

# some of these are fixable once input/output shape info is available to the converter.
# The ones that are not fixed after that should be categorized under "Supported ops, but Unsupported parameters by CoreML"
backend_test.exclude('test_add_bcast_cpu')
backend_test.exclude('test_default_axes_cpu')
backend_test.exclude('test_div_bcast_cpu')
backend_test.exclude('test_flatten_axis0_cpu')
backend_test.exclude('test_logsoftmax_axis_0_cpu')
backend_test.exclude('test_logsoftmax_axis_1_cpu')
backend_test.exclude('test_logsoftmax_axis_2_cpu')
backend_test.exclude('test_logsoftmax_default_axis_cpu')
backend_test.exclude('test_logsoftmax_example_1_cpu')
backend_test.exclude('test_logsoftmax_large_number_cpu')
backend_test.exclude('test_mul_bcast_cpu')
backend_test.exclude('test_slice_cpu')
backend_test.exclude('test_slice_start_out_of_bounds_cpu')
backend_test.exclude('test_softmax_axis_0_cpu')
backend_test.exclude('test_softmax_axis_1_cpu')
backend_test.exclude('test_softmax_axis_2_cpu')
backend_test.exclude('test_softmax_default_axis_cpu')
backend_test.exclude('test_softmax_example_cpu')
backend_test.exclude('test_softmax_large_number_cpu')
backend_test.exclude('test_Softmax_cpu')
backend_test.exclude('test_Softmin_cpu')
backend_test.exclude('test_log_softmax_dim3_cpu')
backend_test.exclude('test_log_softmax_lastdim_cpu')
backend_test.exclude('test_softmax_functional_dim3_cpu')
backend_test.exclude('test_softmax_lastdim_cpu')
backend_test.exclude('test_squeeze_cpu')
backend_test.exclude('test_sub_bcast_cpu')
backend_test.exclude('test_sub_cpu')
backend_test.exclude('test_sub_example_cpu')
backend_test.exclude('test_unsqueeze_cpu')
backend_test.exclude('test_slice_end_out_of_bounds_cpu')
backend_test.exclude('test_slice_neg_cpu')
backend_test.exclude('test_GLU_cpu')
backend_test.exclude('test_GLU_dim_cpu')
backend_test.exclude('test_Linear_cpu')
backend_test.exclude('test_LogSoftmax_cpu')
backend_test.exclude('test_MaxPool1d_cpu')
backend_test.exclude('test_MaxPool1d_stride_cpu')
backend_test.exclude('test_PReLU_1d_multiparam_cpu')
backend_test.exclude('test_operator_chunk_cpu')
backend_test.exclude('test_operator_permute2_cpu')
backend_test.exclude('test_operator_transpose_cpu')

# These layers are supported. Need to fix these tests
backend_test.exclude('test_Softsign_cpu')
backend_test.exclude('test_Upsample_nearest_scale_2d_cpu')
backend_test.exclude('test_operator_maxpool_cpu')
backend_test.exclude('test_operator_params_cpu')


# Unsupported ops by CoreML: error messages should be improved for this category
backend_test.exclude('test_clip_cpu')
backend_test.exclude('test_clip_default_max_cpu')
backend_test.exclude('test_clip_default_min_cpu')
backend_test.exclude('test_clip_example_cpu')
backend_test.exclude('test_operator_clip_cpu')
backend_test.exclude('test_constant_cpu')
backend_test.exclude('test_ceil_cpu')
backend_test.exclude('test_ceil_example_cpu')
backend_test.exclude('test_floor_cpu')
backend_test.exclude('test_floor_example_cpu')
backend_test.exclude('test_hardmax_axis_0_cpu')
backend_test.exclude('test_hardmax_axis_1_cpu')
backend_test.exclude('test_hardmax_axis_2_cpu')
backend_test.exclude('test_hardmax_default_axis_cpu')
backend_test.exclude('test_hardmax_example_cpu')
backend_test.exclude('test_hardmax_one_hot_cpu')
backend_test.exclude('test_matmul_2d_cpu')
backend_test.exclude('test_matmul_3d_cpu')
backend_test.exclude('test_matmul_4d_cpu')
backend_test.exclude('test_AvgPool3d_cpu')
backend_test.exclude('test_AvgPool3d_stride_cpu')
backend_test.exclude('test_BatchNorm3d_eval_cpu')
backend_test.exclude('test_Conv3d_cpu')
backend_test.exclude('test_Conv3d_dilated_cpu')
backend_test.exclude('test_Conv3d_dilated_strided_cpu')
backend_test.exclude('test_Conv3d_groups_cpu')
backend_test.exclude('test_Conv3d_no_bias_cpu')
backend_test.exclude('test_Conv3d_stride_cpu')
backend_test.exclude('test_Conv3d_stride_padding_cpu')
backend_test.exclude('test_MaxPool3d_cpu')
backend_test.exclude('test_MaxPool3d_stride_cpu')
backend_test.exclude('test_MaxPool3d_stride_padding_cpu')
backend_test.exclude('test_PReLU_3d_cpu')
backend_test.exclude('test_PReLU_3d_multiparam_cpu')
backend_test.exclude('test_AvgPool3d_stride1_pad0_gpu_input_cpu')
backend_test.exclude('test_BatchNorm3d_momentum_eval_cpu')
backend_test.exclude('test_BatchNorm1d_3d_input_eval_cpu')

# Supported ops, but Unsupported parameters by CoreML
backend_test.exclude('test_thresholdedrelu_example_cpu')

#maybe these can be fixed if we find a way to convert the "weight" input to a graph initializer field
#otherwise they fall in the category of "supported ops, but unsupported parameters by CoreML"
backend_test.exclude('test_basic_conv_with_padding_cpu')
backend_test.exclude('test_basic_conv_without_padding_cpu')
backend_test.exclude('test_conv_with_strides_and_asymmetric_padding_cpu')
backend_test.exclude('test_conv_with_strides_no_padding_cpu')
backend_test.exclude('test_conv_with_strides_padding_cpu')

#Failing due to to tolerance (in particular due to the way outputs are compared)
backend_test.exclude('test_log_example_cpu')

#These are failing due to some error in Caffe2 backend
backend_test.exclude('test_hardsigmoid_cpu')
backend_test.exclude('test_hardsigmoid_default_cpu')
backend_test.exclude('test_hardsigmoid_example_cpu')
backend_test.exclude('test_mean_example_cpu')
backend_test.exclude('test_mean_one_input_cpu')
backend_test.exclude('test_mean_two_inputs_cpu')
backend_test.exclude('test_pow_bcast_axis0_cpu')
backend_test.exclude('test_pow_bcast_cpu')
backend_test.exclude('test_pow_cpu')
backend_test.exclude('test_pow_example_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT16_cpu')
backend_test.exclude('test_cast_FLOAT16_to_DOUBLE_cpu')
backend_test.exclude('test_cast_FLOAT16_to_FLOAT_cpu')
backend_test.exclude('test_cast_FLOAT_to_FLOAT16_cpu')

# These fail due to "TypeError: Input must be of of type TensorProto.FLOAT" or
# "TypeError: Output must be of of type TensorProto.FLOAT"
# Some of these can be fixed by forcing non float types to be float types in CoreML
backend_test.exclude('test_gather_1_cpu')
backend_test.exclude('test_gather_0_cpu')
backend_test.exclude('test_and2d_cpu')
backend_test.exclude('test_and3d_cpu')
backend_test.exclude('test_and4d_cpu')
backend_test.exclude('test_and_axis0_cpu')
backend_test.exclude('test_and_axis1_cpu')
backend_test.exclude('test_and_axis2_cpu')
backend_test.exclude('test_and_axis3_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT_cpu')
backend_test.exclude('test_concat_1d_axis_0_cpu')
backend_test.exclude('test_concat_2d_axis_0_cpu')
backend_test.exclude('test_concat_2d_axis_1_cpu')
backend_test.exclude('test_concat_3d_axis_0_cpu')
backend_test.exclude('test_concat_3d_axis_1_cpu')
backend_test.exclude('test_concat_3d_axis_2_cpu')
backend_test.exclude('test_equal_bcast_cpu')
backend_test.exclude('test_equal_cpu')
backend_test.exclude('test_not_2d_cpu')
backend_test.exclude('test_not_3d_cpu')
backend_test.exclude('test_not_4d_cpu')
backend_test.exclude('test_or2d_cpu')
backend_test.exclude('test_or3d_cpu')
backend_test.exclude('test_or4d_cpu')
backend_test.exclude('test_or_axis0_cpu')
backend_test.exclude('test_or_axis1_cpu')
backend_test.exclude('test_or_axis2_cpu')
backend_test.exclude('test_or_axis3_cpu')
backend_test.exclude('test_or_bcast3v1d_cpu')
backend_test.exclude('test_or_bcast3v2d_cpu')
backend_test.exclude('test_or_bcast4v2d_cpu')
backend_test.exclude('test_or_bcast4v3d_cpu')
backend_test.exclude('test_xor2d_cpu')
backend_test.exclude('test_xor3d_cpu')
backend_test.exclude('test_xor4d_cpu')
backend_test.exclude('test_xor_axis0_cpu')
backend_test.exclude('test_xor_axis1_cpu')
backend_test.exclude('test_xor_axis2_cpu')
backend_test.exclude('test_xor_axis3_cpu')
backend_test.exclude('test_xor_bcast3v1d_cpu')
backend_test.exclude('test_xor_bcast3v2d_cpu')
backend_test.exclude('test_xor_bcast4v2d_cpu')
backend_test.exclude('test_xor_bcast4v3d_cpu')
backend_test.exclude('test_Embedding_cpu')
backend_test.exclude('test_Embedding_sparse_cpu')
backend_test.exclude('test_operator_equal_cpu')
backend_test.exclude('test_operator_non_float_params_cpu')
backend_test.exclude('test_cast_FLOAT_to_DOUBLE_cpu')
backend_test.exclude('test_greater_bcast_cpu')
backend_test.exclude('test_greater_cpu')
backend_test.exclude('test_less_bcast_cpu')
backend_test.exclude('test_less_cpu')
backend_test.exclude('test_shape_cpu')
backend_test.exclude('test_shape_example_cpu')
backend_test.exclude('test_size_cpu')
backend_test.exclude('test_size_example_cpu')
backend_test.exclude('test_top_k_cpu')


# exclude all the model zoo tests. They are tested elsewhere.
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
