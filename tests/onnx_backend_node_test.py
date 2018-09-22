from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx
from typing import Any, Text, Optional, Tuple, Sequence
import onnx.backend.test
import numpy

from onnx_coreml._backend import CoreMLBackend

class CoreMLTestingBackend(CoreMLBackend):
    @classmethod
    def run_node(cls,
                 node,  # type: onnx.NodeProto
                 inputs,  # type: Sequence[numpy.ndarray[Any]]
                 device='CPU',  # type: Text
                 outputs_info=None,  # type: Optional[Sequence[Tuple[numpy.dtype, Tuple[int, ...]]]]
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

        assert outputs_info is not None, "CoreML needs output shapes"

        graph_inputs = []
        for i in range(len(inputs)):
            input_ = inputs[i]
            value_info = onnx.helper.make_tensor_value_info(
                name=node.input[i],
                elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[input_.dtype],
                shape=input_.shape
            )
            graph_inputs.append(value_info)

        graph_outputs = []
        for i in range(len(outputs_info)):
            output_info = outputs_info[i]
            value_info = onnx.helper.make_tensor_value_info(
                name=node.output[i],
                elem_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[output_info[0]],
                shape=output_info[1]
            )
            graph_outputs.append(value_info)

        graph = onnx.helper.make_graph(
            nodes=[node],
            name='dummy',
            inputs=graph_inputs,
            outputs=graph_outputs,
        )

        model = onnx.helper.make_model(graph)
        return cls.prepare(model).run(inputs)


# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(CoreMLTestingBackend, __name__)

# Failing due to tolerance (in particular due to the way outputs are compared)
backend_test.exclude('test_log_example_cpu')
backend_test.exclude('test_ConvTranspose2d_cpu') #onnx ref out does not match with what the document states: output pad should be zero pad
backend_test.exclude('test_ConvTranspose2d_no_bias_cpu')  # same as above
backend_test.exclude('test_operator_convtranspose_cpu') # same as above

# These are failing due to some error in Caffe2 backend
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
backend_test.exclude('test_depthtospace_cpu')
backend_test.exclude('test_depthtospace_example_cpu')
backend_test.exclude('test_PixelShuffle_cpu') #error in shape inference

# Failure due to onnx
backend_test.exclude('test_BatchNorm2d_eval_cpu')
backend_test.exclude('test_BatchNorm2d_momentum_eval_cpu')
backend_test.exclude('test_reshape_extended_dims_cpu')
backend_test.exclude('test_reshape_negative_dim_cpu')
backend_test.exclude('test_reshape_one_dim_cpu')
backend_test.exclude('test_reshape_reduced_dims_cpu')
backend_test.exclude('test_reshape_reordered_dims_cpu')

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
backend_test.exclude('test_sub_bcast_cpu')
backend_test.exclude('test_sub_cpu')
backend_test.exclude('test_sub_example_cpu')
backend_test.exclude('test_slice_end_out_of_bounds_cpu')
backend_test.exclude('test_slice_neg_cpu')
backend_test.exclude('test_GLU_cpu')
backend_test.exclude('test_GLU_dim_cpu')
backend_test.exclude('test_Linear_cpu')
backend_test.exclude('test_LogSoftmax_cpu')
backend_test.exclude('test_PReLU_1d_multiparam_cpu')
backend_test.exclude('test_operator_chunk_cpu')
backend_test.exclude('test_operator_permute2_cpu')
backend_test.exclude('test_operator_transpose_cpu')
backend_test.exclude('test_slice_default_axes_cpu')
backend_test.exclude('test_gather_1_cpu')
backend_test.exclude('test_gather_0_cpu')
backend_test.exclude('test_Embedding_cpu')
backend_test.exclude('test_Embedding_sparse_cpu')
backend_test.exclude('test_operator_add_broadcast_cpu')
backend_test.exclude('test_operator_add_size1_broadcast_cpu')
backend_test.exclude('test_operator_add_size1_right_broadcast_cpu')
backend_test.exclude('test_operator_add_size1_singleton_broadcast_cpu')
backend_test.exclude('test_operator_index_cpu')
backend_test.exclude('test_operator_pow_cpu')
backend_test.exclude('test_concat_2d_axis_0_cpu')
backend_test.exclude('test_argmax_default_axis_example_cpu')
backend_test.exclude('test_argmin_default_axis_example_cpu')
backend_test.exclude('test_argmax_no_keepdims_example_cpu')
backend_test.exclude('test_argmin_no_keepdims_example_cpu')

backend_test.exclude('test_mvn_cpu')
backend_test.exclude('test_maxpool_with_argmax_2d_precomputed_pads_cpu')
backend_test.exclude('test_maxpool_with_argmax_2d_precomputed_strides_cpu')

# 1-D conv/pool cases. an be supported using shape inference
backend_test.exclude('test_Conv1d_cpu')
backend_test.exclude('test_Conv1d_dilated_cpu')
backend_test.exclude('test_Conv1d_groups_cpu')
backend_test.exclude('test_Conv1d_pad1_cpu')
backend_test.exclude('test_Conv1d_pad1size1_cpu')
backend_test.exclude('test_Conv1d_pad2_cpu')
backend_test.exclude('test_Conv1d_pad2size1_cpu')
backend_test.exclude('test_Conv1d_stride_cpu')
backend_test.exclude('test_MaxPool1d_cpu')
backend_test.exclude('test_MaxPool1d_stride_cpu')
backend_test.exclude('test_averagepool_1d_default_cpu')
backend_test.exclude('test_maxpool_1d_default_cpu')
backend_test.exclude('test_AvgPool1d_cpu')
backend_test.exclude('test_AvgPool1d_stride_cpu')

# recurrent tests.
backend_test.exclude('test_operator_rnn_cpu')
backend_test.exclude('test_operator_rnn_single_layer_cpu')
backend_test.exclude('test_operator_lstm_cpu')
backend_test.exclude('test_gru_defaults_cpu')
backend_test.exclude('test_gru_with_initial_bias_cpu')
backend_test.exclude('test_lstm_defaults_cpu')
backend_test.exclude('test_lstm_with_initial_bias_cpu')
backend_test.exclude('test_lstm_with_peepholes_cpu')
backend_test.exclude('test_simple_rnn_defaults_cpu')
backend_test.exclude('test_simple_rnn_with_initial_bias_cpu')


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
backend_test.exclude('test_averagepool_3d_default_cpu')
backend_test.exclude('test_maxpool_3d_default_cpu')
backend_test.exclude('test_split_variable_parts_1d_cpu')
backend_test.exclude('test_split_variable_parts_2d_cpu')
backend_test.exclude('test_split_variable_parts_default_axis_cpu')
backend_test.exclude('test_cast_DOUBLE_to_FLOAT_cpu')
backend_test.exclude('test_cast_FLOAT_to_DOUBLE_cpu')
backend_test.exclude('test_shape_cpu')
backend_test.exclude('test_shape_example_cpu')
backend_test.exclude('test_size_cpu')
backend_test.exclude('test_size_example_cpu')
backend_test.exclude('test_top_k_cpu')
backend_test.exclude('test_concat_3d_axis_1_cpu')
backend_test.exclude('test_concat_3d_axis_2_cpu')
backend_test.exclude('test_Linear_no_bias_cpu')
backend_test.exclude('test_PoissonNLLLLoss_no_reduce_cpu')
backend_test.exclude('test_operator_addconstant_cpu')
backend_test.exclude('test_operator_mm_cpu')
backend_test.exclude('test_operator_addmm_cpu')
backend_test.exclude('test_operator_type_as_cpu')
backend_test.exclude('test_tile_cpu')
backend_test.exclude('test_tile_precomputed_cpu')
backend_test.exclude('test_acos_cpu')
backend_test.exclude('test_acos_example_cpu')
backend_test.exclude('test_asin_cpu')
backend_test.exclude('test_asin_example_cpu')
backend_test.exclude('test_atan_cpu')
backend_test.exclude('test_atan_example_cpu')
backend_test.exclude('test_cos_cpu')
backend_test.exclude('test_cos_example_cpu')
backend_test.exclude('test_sin_cpu')
backend_test.exclude('test_sin_example_cpu')
backend_test.exclude('test_tan_cpu')
backend_test.exclude('test_tan_example_cpu')
backend_test.exclude('test_dropout_cpu')
backend_test.exclude('test_dropout_default_cpu')
backend_test.exclude('test_dropout_random_cpu')
backend_test.exclude('test_gru_seq_length_cpu')
backend_test.exclude('test_identity_cpu')
backend_test.exclude('test_pow_bcast_scalar_cpu')
backend_test.exclude('test_pow_bcast_array_cpu')
backend_test.exclude('test_asin_example_cpu')
backend_test.exclude('test_reduce_log_sum_exp_default_axes_keepdims_example_cpu')
backend_test.exclude('test_reduce_log_sum_exp_default_axes_keepdims_random_cpu')
backend_test.exclude('test_reduce_log_sum_exp_do_not_keepdims_example_cpu')
backend_test.exclude('test_reduce_log_sum_exp_do_not_keepdims_random_cpu')
backend_test.exclude('test_reduce_log_sum_exp_keepdims_example_cpu')
backend_test.exclude('test_reduce_log_sum_exp_keepdims_random_cpu')
backend_test.exclude('test_rnn_seq_length_cpu')
backend_test.exclude('test_operator_repeat_cpu')
backend_test.exclude('test_operator_repeat_dim_overflow_cpu')
backend_test.exclude('test_reduce_log_sum_default_cpu')
backend_test.exclude('test_thresholdedrelu_example_cpu') #different convention for CoreML
backend_test.exclude('test_expand_shape_model1_cpu')
backend_test.exclude('test_expand_shape_model2_cpu')
backend_test.exclude('test_expand_shape_model3_cpu')
backend_test.exclude('test_expand_shape_model4_cpu')
backend_test.exclude('test_expand_dim_changed_cpu')
backend_test.exclude('test_expand_dim_unchanged_cpu')

# Supported ops, but ONNX passes in dynamic inputs that CoreML needs to be constant graph initializers
backend_test.exclude('test_gemm_broadcast_cpu')
backend_test.exclude('test_gemm_nobroadcast_cpu')


# CoreML doesn't support tensor layout
backend_test.exclude('test_reduce_log_sum_desc_axes_cpu')
backend_test.exclude('test_reduce_log_sum_cpu')
backend_test.exclude('test_reduce_log_sum_asc_axes_cpu')


# Unsupported input data type and op
backend_test.exclude('test_and2d_cpu')
backend_test.exclude('test_and3d_cpu')
backend_test.exclude('test_and4d_cpu')
backend_test.exclude('test_and_axis0_cpu')
backend_test.exclude('test_and_axis1_cpu')
backend_test.exclude('test_and_axis2_cpu')
backend_test.exclude('test_and_axis3_cpu')
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
backend_test.exclude('test_operator_equal_cpu')
backend_test.exclude('test_operator_non_float_params_cpu')
backend_test.exclude('test_greater_bcast_cpu')
backend_test.exclude('test_greater_cpu')
backend_test.exclude('test_less_bcast_cpu')
backend_test.exclude('test_less_cpu')
backend_test.exclude('test_and_bcast3v1d_cpu')
backend_test.exclude('test_and_bcast3v2d_cpu')
backend_test.exclude('test_and_bcast4v2d_cpu')
backend_test.exclude('test_and_bcast4v3d_cpu')
backend_test.exclude('test_and_bcast4v4d_cpu')
backend_test.exclude('test_or_bcast4v4d_cpu')
backend_test.exclude('test_xor_bcast4v4d_cpu')


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
backend_test.exclude('test_zfnet')





globals().update(backend_test
                 .enable_report()
                 .test_cases)


if __name__ == '__main__':
    unittest.main()

