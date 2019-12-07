from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx
from typing import Any, Text, Optional, Tuple, Sequence
import onnx.backend.test
import numpy

from onnx_coreml._backend import CoreMLBackend, CoreMLBackendND
from onnx_coreml.converter import SupportedVersion

from coremltools.models.utils import macos_version

# Default target iOS
MINIMUM_IOS_DEPLOYMENT_TARGET = '13'

MIN_MACOS_VERSION_10_15 = (10, 15)
# If MACOS version is less than 10.15
# Then force testing on CoreML 2.0
if macos_version() < MIN_MACOS_VERSION_10_15:
    MINIMUM_IOS_DEPLOYMENT_TARGET = '12'

if not SupportedVersion.ios_support_check(MINIMUM_IOS_DEPLOYMENT_TARGET):
    raise ValueError(
        "Invalid Target iOS version provided. Valid target iOS: {}".format(supported_ios_version)
    )

class CoreMLTestingBackend(CoreMLBackendND if SupportedVersion.is_nd_array_supported(MINIMUM_IOS_DEPLOYMENT_TARGET) else CoreMLBackend):
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


def exclude_test_cases(backend_test):
    # Test cases to be disabled till coreml 2
    unsupported_tests_till_coreml2 = [
        # Failing due to tolerance (in particular due to the way outputs are compared,
        'test_log_example_cpu',
        'test_operator_add_broadcast_cpu',
        'test_operator_add_size1_right_broadcast_cpu',
        'test_operator_add_size1_singleton_broadcast_cpu',
        'test_operator_addconstant_cpu',
        'test_sign_cpu',
        'test_sign_model_cpu',

        # Dynamic ONNX ops not supported in CoreML
        'test_reshape_extended_dims_cpu',
        'test_reshape_negative_dim_cpu',
        'test_reshape_one_dim_cpu',
        'test_reshape_reduced_dims_cpu',
        'test_reshape_reordered_dims_cpu',
        'test_gemm_broadcast_cpu',
        'test_gemm_nobroadcast_cpu',
        'test_upsample_nearest_cpu',

        # Failure due to axis alignment between ONNX and CoreML
        'test_add_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_div_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_mul_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_sub_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_operator_index_cpu', # input: [1,1]: cannot slice along batch axis
        'test_concat_2d_axis_0_cpu', # cannot slice along batch axis
        'test_argmax_default_axis_example_cpu', # batch axis
        'test_argmin_default_axis_example_cpu', # batch axis
        'test_AvgPool1d_cpu', # 3-D tensor unsqueezed to 4D with axis = 3 and then pool
        'test_AvgPool1d_stride_cpu', # same as above

        # Possibly Fixable by the converter
        'test_slice_start_out_of_bounds_cpu', # starts and ends exceed input size

        # "Supported ops, but Unsupported parameters by CoreML"
        'test_logsoftmax_axis_1_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_logsoftmax_default_axis_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_softmax_axis_1_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_softmax_default_axis_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_logsoftmax_axis_0_cpu',
        'test_logsoftmax_axis_2_cpu',
        'test_softmax_axis_0_cpu',
        'test_softmax_axis_2_cpu',
        'test_log_softmax_dim3_cpu',
        'test_log_softmax_lastdim_cpu',
        'test_softmax_functional_dim3_cpu',
        'test_PReLU_1d_multiparam_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_mvn_cpu', # not sure why there is numerical mismatch
        'test_flatten_axis2_cpu', # 2,3,4,5 --> 6,20
        'test_flatten_axis3_cpu', # 2,3,4,5 --> 24,5

        # Unsupported ops by CoreML
        'test_constant_cpu',
        'test_hardmax_axis_0_cpu',
        'test_hardmax_axis_1_cpu',
        'test_hardmax_axis_2_cpu',
        'test_hardmax_default_axis_cpu',
        'test_hardmax_example_cpu',
        'test_hardmax_one_hot_cpu',
        'test_matmul_2d_cpu',
        'test_matmul_3d_cpu',
        'test_matmul_4d_cpu',
        'test_AvgPool3d_cpu',
        'test_AvgPool3d_stride_cpu',
        'test_BatchNorm3d_eval_cpu',
        'test_Conv3d_cpu',
        'test_Conv3d_dilated_cpu',
        'test_Conv3d_dilated_strided_cpu',
        'test_Conv3d_groups_cpu',
        'test_Conv3d_no_bias_cpu',
        'test_Conv3d_stride_cpu',
        'test_Conv3d_stride_padding_cpu',
        'test_MaxPool3d_cpu',
        'test_MaxPool3d_stride_cpu',
        'test_MaxPool3d_stride_padding_cpu',
        'test_PReLU_3d_cpu',
        'test_PReLU_3d_multiparam_cpu',
        'test_AvgPool3d_stride1_pad0_gpu_input_cpu',
        'test_BatchNorm3d_momentum_eval_cpu',
        'test_BatchNorm1d_3d_input_eval_cpu',
        'test_averagepool_3d_default_cpu',
        'test_maxpool_3d_default_cpu',
        'test_split_variable_parts_1d_cpu',
        'test_split_variable_parts_2d_cpu',
        'test_split_variable_parts_default_axis_cpu',
        'test_cast_DOUBLE_to_FLOAT_cpu',
        'test_cast_FLOAT_to_DOUBLE_cpu',
        'test_shape_cpu',
        'test_shape_example_cpu',
        'test_size_cpu',
        'test_size_example_cpu',
        'test_top_k_cpu',
        'test_PoissonNLLLLoss_no_reduce_cpu',
        'test_operator_mm_cpu',
        'test_operator_addmm_cpu',
        'test_tile_cpu',
        'test_tile_precomputed_cpu',
        'test_acos_cpu',
        'test_acos_example_cpu',
        'test_asin_cpu',
        'test_asin_example_cpu',
        'test_atan_cpu',
        'test_atan_example_cpu',
        'test_cos_cpu',
        'test_cos_example_cpu',
        'test_sin_cpu',
        'test_sin_example_cpu',
        'test_tan_cpu',
        'test_tan_example_cpu',
        'test_dropout_cpu',
        'test_dropout_default_cpu',
        'test_dropout_random_cpu',
        'test_gru_seq_length_cpu',
        'test_identity_cpu',
        'test_asin_example_cpu',
        'test_reduce_log_sum_exp_default_axes_keepdims_example_cpu',
        'test_reduce_log_sum_exp_default_axes_keepdims_random_cpu',
        'test_reduce_log_sum_exp_do_not_keepdims_example_cpu',
        'test_reduce_log_sum_exp_do_not_keepdims_random_cpu',
        'test_reduce_log_sum_exp_keepdims_example_cpu',
        'test_reduce_log_sum_exp_keepdims_random_cpu',
        'test_rnn_seq_length_cpu',
        'test_operator_repeat_cpu',
        'test_operator_repeat_dim_overflow_cpu',
        'test_thresholdedrelu_example_cpu', #different convention for CoreML
        'test_expand_shape_model1_cpu',
        'test_expand_shape_model2_cpu',
        'test_expand_shape_model3_cpu',
        'test_expand_shape_model4_cpu',
        'test_expand_dim_changed_cpu',
        'test_expand_dim_unchanged_cpu',
        'test_operator_chunk_cpu', # unequal splits
        'test_operator_permute2_cpu', # rank 6 input
        'test_maxpool_with_argmax_2d_precomputed_pads_cpu',
        'test_maxpool_with_argmax_2d_precomputed_strides_cpu',
        'test_gather_1_cpu',
        'test_gather_0_cpu',
        'test_Embedding_cpu', # gather op
        'test_Embedding_sparse_cpu', # gather op
        'test_pow_bcast_scalar_cpu',
        'test_pow_bcast_array_cpu',
        'test_pow_cpu',
        'test_pow_example_cpu',
        'test_operator_pow_cpu',
        'test_shrink_cpu',
        'test_acosh_cpu',
        'test_asinh_cpu',
        'test_atanh_cpu',
        'test_erf_cpu',
        'test_isnan_cpu',
        'test_where_example_cpu',
        'test_acosh_example_cpu',
        'test_asinh_example_cpu',
        'test_atanh_example_cpu',
        'test_compress_0_cpu',
        'test_compress_1_cpu',
        'test_compress_default_axis_cpu',
        'test_constantofshape_float_ones_cpu',
        'test_constantofshape_int_zeros_cpu',
        'test_cosh_cpu',
        'test_cosh_example_cpu',
        'test_dynamic_slice_cpu',
        'test_dynamic_slice_default_axes_cpu',
        'test_dynamic_slice_end_out_of_bounds_cpu',
        'test_dynamic_slice_neg_cpu',
        'test_dynamic_slice_neg_cpu',
        'test_dynamic_slice_start_out_of_bounds_cpu',
        'test_eyelike_populate_off_main_diagonal_cpu',
        'test_eyelike_with_dtype_cpu',
        'test_eyelike_without_dtype_cpu',
        'test_maxunpool_export_with_output_shape_cpu',
        'test_maxunpool_export_without_output_shape_cpu',
        'test_nonzero_example_cpu',
        'test_onehot_with_axis_cpu',
        'test_onehot_without_axis_cpu',
        'test_scan9_sum_cpu',
        'test_scan_sum_cpu',
        'test_scatter_with_axis_cpu',
        'test_scatter_without_axis_cpu',
        'test_shrink_hard_cpu',
        'test_shrink_soft_cpu',
        'test_sinh_cpu',
        'test_sinh_example_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_only_bigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu',
        'test_basic_convinteger_cpu',


        # recurrent tests.
        'test_operator_rnn_cpu',
        'test_operator_rnn_single_layer_cpu',
        'test_operator_lstm_cpu',
        'test_gru_defaults_cpu',
        'test_gru_with_initial_bias_cpu',
        'test_lstm_defaults_cpu',
        'test_lstm_with_initial_bias_cpu',
        'test_lstm_with_peepholes_cpu',
        'test_simple_rnn_defaults_cpu',
        'test_simple_rnn_with_initial_bias_cpu',

        # exclude all the model zoo tests. They are tested elsewhere.
        'test_bvlc_alexnet',
        'test_resnet50',
        'test_vgg16',
        'test_vgg19',
        'test_densenet121',
        'test_inception_v1',
        'test_inception_v2',
        'test_shufflenet',
        'test_squeezenet',
        'test_zfnet',
        'test_maxunpool_export_with_output_shape_cpu',
        'test_maxunpool_export_without_output_shape_cpu',
        'test_nonzero_example_cpu',
        'test_onehot_with_axis_cpu',
        'test_onehot_without_axis_cpu',
        'test_scan9_sum_cpu',
        'test_scan_sum_cpu',
        'test_shrink_hard_cpu',
        'test_shrink_soft_cpu',
        'test_sinh_cpu',
        'test_sinh_example_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_only_bigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu',

        # Incorrect test setup
        'test_mod_float_mixed_sign_example_cpu',
        'test_mod_int64_mixed_sign_example_cpu',
        
        # Need dead code elimination and removing unused inputs
        'test_batchnorm_epsilon_cpu',
        'test_batchnorm_example_cpu',

        # recurrent tests.
        'test_operator_rnn_cpu',
        'test_operator_rnn_single_layer_cpu',
        'test_operator_lstm_cpu',
        'test_gru_defaults_cpu',
        'test_gru_with_initial_bias_cpu',
        'test_lstm_defaults_cpu',
        'test_lstm_with_initial_bias_cpu',
        'test_lstm_with_peepholes_cpu',
        'test_simple_rnn_defaults_cpu',
        'test_simple_rnn_with_initial_bias_cpu',

        # exclude all the model zoo tests. They are tested elsewhere.
        'test_bvlc_alexnet',
        'test_resnet50',
        'test_vgg16',
        'test_vgg19',
        'test_densenet121',
        'test_inception_v1',
        'test_inception_v2',
        'test_shufflenet',
        'test_squeezenet',
        'test_zfnet'
        'test_maxunpool_export_with_output_shape_cpu',
        'test_maxunpool_export_without_output_shape_cpu',
        'test_nonzero_example_cpu',
        'test_onehot_with_axis_cpu',
        'test_onehot_without_axis_cpu',
        'test_scan9_sum_cpu',
        'test_scan_sum_cpu',
        'test_shrink_hard_cpu',
        'test_shrink_soft_cpu',
        'test_sinh_cpu',
        'test_sinh_example_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_only_bigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu',
        'test_basic_convinteger_cpu',
        'test_cast_FLOAT_to_STRING_cpu',
        'test_cast_STRING_to_FLOAT_cpu',
        'test_cast_DOUBLE_to_FLOAT16_cpu',
        'test_cast_FLOAT16_to_DOUBLE_cpu',
        'test_cast_FLOAT16_to_FLOAT_cpu',
        'test_cast_FLOAT_to_FLOAT16_cpu',
        'test_matmulinteger_cpu',
        'test_prelu_broadcast_cpu',
        'test_convinteger_with_padding_cpu',
        'test_convtranspose_3d_cpu',
        'test_dequantizelinear_cpu',
        'test_instancenorm_epsilon_cpu',
        'test_instancenorm_example_cpu',
        'test_isinf_cpu',
        'test_isinf_negative_cpu',
        'test_isinf_positive_cpu',
        'test_nonmaxsuppression_center_point_box_format_cpu',
        'test_nonmaxsuppression_flipped_coordinates_cpu',
        'test_nonmaxsuppression_identical_boxes_cpu',
        'test_nonmaxsuppression_limit_output_size_cpu',
        'test_nonmaxsuppression_single_box_cpu',
        'test_nonmaxsuppression_suppress_by_IOU_and_scores_cpu',
        'test_nonmaxsuppression_suppress_by_IOU_cpu',
        'test_nonmaxsuppression_two_batches_cpu',
        'test_nonmaxsuppression_two_classes_cpu',
        'test_prelu_example_cpu',
        'test_qlinearconv_cpu',
        'test_qlinearmatmul_2D_cpu',
        'test_qlinearmatmul_3D_cpu',
        'test_quantizelinear_cpu',
        'test_resize_downsample_linear_cpu',
        'test_resize_downsample_nearest_cpu',
        'test_resize_nearest_cpu',
        'test_resize_upsample_linear_cpu',
        'test_resize_upsample_nearest_cpu',
        'test_reversesequence_batch_cpu',
        'test_reversesequence_time_cpu',
        'test_roialign_cpu',
        'test_slice_cpu',
        'test_slice_default_axes_cpu',
        'test_slice_default_steps_cpu',
        'test_slice_end_out_of_bounds_cpu',
        'test_slice_neg_cpu',
        'test_slice_neg_steps_cpu',
        'test_strnormalizer_export_monday_casesensintive_lower_cpu',
        'test_strnormalizer_export_monday_casesensintive_nochangecase_cpu',
        'test_strnormalizer_export_monday_casesensintive_upper_cpu',
        'test_strnormalizer_export_monday_empty_output_cpu',
        'test_strnormalizer_export_monday_insensintive_upper_twodim_cpu',
        'test_strnormalizer_nostopwords_nochangecase_cpu',
        'test_PReLU_1d_cpu',
        'test_PReLU_2d_cpu',
        'test_strnorm_model_monday_casesensintive_lower_cpu',
        'test_strnorm_model_monday_casesensintive_nochangecase_cpu',
        'test_strnorm_model_monday_casesensintive_upper_cpu',
        'test_strnorm_model_monday_empty_output_cpu',
        'test_strnorm_model_monday_insensintive_upper_twodim_cpu',
        'test_strnorm_model_nostopwords_nochangecase_cpu',
        'test_averagepool_2d_ceil_cpu',
        'test_convtranspose_pad_cpu',
        'test_logsoftmax_large_number_cpu',
        'test_maxpool_2d_ceil_cpu',
        'test_maxpool_2d_dilations_cpu',
        'test_softmax_large_number_cpu',
        'test_mod_bcast_cpu',
        'test_mod_fmod_mixed_sign_example_cpu',
        'test_mvn_expanded_cpu',
        'test_operator_non_float_params_cpu',
        'test_operator_params_cpu',
    ]

    ## Test cases to be disabled in CoreML 3.0
    unsupported_tests_coreml3 = [
        # Failing due to tolerance (in particular due to the way outputs are compared,
        'test_log_example_cpu',
        'test_operator_add_broadcast_cpu',
        'test_operator_add_size1_right_broadcast_cpu',
        'test_operator_add_size1_singleton_broadcast_cpu',
        'test_operator_addconstant_cpu',
        'test_sign_cpu',
        'test_sign_model_cpu',

        # Dynamic ONNX ops not supported in CoreML
        'test_reshape_extended_dims_cpu',
        'test_reshape_negative_dim_cpu',
        'test_reshape_one_dim_cpu',
        'test_reshape_reduced_dims_cpu',
        'test_reshape_reordered_dims_cpu',
        'test_gemm_broadcast_cpu',
        'test_gemm_nobroadcast_cpu',
        'test_upsample_nearest_cpu',

        # Failure due to axis alignment between ONNX and CoreML
        'test_add_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_div_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_mul_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_sub_bcast_cpu', # (3,4,5, shaped tensor + (5,, shaped tensor
        'test_operator_index_cpu', # input: [1,1]: cannot slice along batch axis
        'test_concat_2d_axis_0_cpu', # cannot slice along batch axis
        'test_argmax_default_axis_example_cpu', # batch axis
        'test_argmin_default_axis_example_cpu', # batch axis
        'test_AvgPool1d_cpu', # 3-D tensor unsqueezed to 4D with axis = 3 and then pool
        'test_AvgPool1d_stride_cpu', # same as above

        # Possibly Fixable by the converter
        'test_slice_start_out_of_bounds_cpu', # starts and ends exceed input size

        # "Supported ops, but Unsupported parameters by CoreML"
        'test_logsoftmax_axis_1_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_logsoftmax_default_axis_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_softmax_axis_1_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_softmax_default_axis_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_logsoftmax_axis_0_cpu',
        'test_softmax_axis_0_cpu',
        'test_PReLU_1d_multiparam_cpu', # this one converts but fails at runtime: must give error during conversion
        'test_mvn_cpu', # not sure why there is numerical mismatch
        'test_flatten_axis2_cpu', # 2,3,4,5 --> 6,20
        'test_flatten_axis3_cpu', # 2,3,4,5 --> 24,5

        # Unsupported ops by CoreML
        'test_constant_cpu',
        'test_hardmax_axis_0_cpu',
        'test_hardmax_axis_1_cpu',
        'test_hardmax_axis_2_cpu',
        'test_hardmax_default_axis_cpu',
        'test_hardmax_example_cpu',
        'test_hardmax_one_hot_cpu',
        'test_matmul_2d_cpu',
        'test_matmul_3d_cpu',
        'test_matmul_4d_cpu',
        'test_AvgPool3d_cpu',
        'test_AvgPool3d_stride_cpu',
        'test_BatchNorm3d_eval_cpu',
        'test_Conv3d_cpu',
        'test_Conv3d_dilated_cpu',
        'test_Conv3d_dilated_strided_cpu',
        'test_Conv3d_groups_cpu',
        'test_Conv3d_no_bias_cpu',
        'test_Conv3d_stride_cpu',
        'test_Conv3d_stride_padding_cpu',
        'test_MaxPool3d_cpu',
        'test_MaxPool3d_stride_cpu',
        'test_MaxPool3d_stride_padding_cpu',
        'test_PReLU_3d_cpu',
        'test_PReLU_3d_multiparam_cpu',
        'test_AvgPool3d_stride1_pad0_gpu_input_cpu',
        'test_BatchNorm3d_momentum_eval_cpu',
        'test_BatchNorm1d_3d_input_eval_cpu',
        'test_averagepool_3d_default_cpu',
        'test_maxpool_3d_default_cpu',
        'test_split_variable_parts_1d_cpu',
        'test_split_variable_parts_2d_cpu',
        'test_split_variable_parts_default_axis_cpu',
        'test_shape_cpu',
        'test_shape_example_cpu',
        'test_top_k_cpu',
        'test_PoissonNLLLLoss_no_reduce_cpu',
        'test_operator_mm_cpu',
        'test_operator_addmm_cpu',
        'test_tile_cpu',
        'test_tile_precomputed_cpu',
        'test_acos_cpu',
        'test_acos_example_cpu',
        'test_asin_cpu',
        'test_asin_example_cpu',
        'test_atan_cpu',
        'test_atan_example_cpu',
        'test_cos_cpu',
        'test_cos_example_cpu',
        'test_sin_cpu',
        'test_sin_example_cpu',
        'test_tan_cpu',
        'test_tan_example_cpu',
        'test_dropout_cpu',
        'test_dropout_default_cpu',
        'test_dropout_random_cpu',
        'test_gru_seq_length_cpu',
        'test_identity_cpu',
        'test_asin_example_cpu',
        'test_reduce_log_sum_exp_default_axes_keepdims_example_cpu',
        'test_reduce_log_sum_exp_default_axes_keepdims_random_cpu',
        'test_reduce_log_sum_exp_do_not_keepdims_example_cpu',
        'test_reduce_log_sum_exp_do_not_keepdims_random_cpu',
        'test_reduce_log_sum_exp_keepdims_example_cpu',
        'test_reduce_log_sum_exp_keepdims_random_cpu',
        'test_rnn_seq_length_cpu',
        'test_operator_repeat_cpu',
        'test_operator_repeat_dim_overflow_cpu',
        'test_thresholdedrelu_example_cpu', #different convention for CoreML
        'test_expand_shape_model1_cpu',
        'test_expand_shape_model2_cpu',
        'test_expand_shape_model3_cpu',
        'test_expand_shape_model4_cpu',
        'test_expand_dim_changed_cpu',
        'test_operator_chunk_cpu', # unequal splits
        'test_operator_permute2_cpu', # rank 6 input
        'test_maxpool_with_argmax_2d_precomputed_pads_cpu',
        'test_maxpool_with_argmax_2d_precomputed_strides_cpu',
        'test_gather_1_cpu',
        'test_gather_0_cpu',
        'test_Embedding_cpu', # gather op
        'test_Embedding_sparse_cpu', # gather op
        'test_pow_bcast_scalar_cpu',
        'test_pow_bcast_array_cpu',
        'test_pow_cpu',
        'test_pow_example_cpu',
        'test_operator_pow_cpu',
        'test_shrink_cpu',
        'test_acosh_cpu',
        'test_asinh_cpu',
        'test_atanh_cpu',
        'test_isnan_cpu',
        'test_acosh_example_cpu',
        'test_asinh_example_cpu',
        'test_atanh_example_cpu',
        'test_compress_0_cpu',
        'test_compress_1_cpu',
        'test_compress_default_axis_cpu',
        'test_constantofshape_float_ones_cpu',
        'test_constantofshape_int_zeros_cpu',
        'test_cosh_cpu',
        'test_cosh_example_cpu',
        'test_dynamic_slice_cpu',
        'test_dynamic_slice_default_axes_cpu',
        'test_dynamic_slice_end_out_of_bounds_cpu',
        'test_dynamic_slice_neg_cpu',
        'test_dynamic_slice_neg_cpu',
        'test_dynamic_slice_start_out_of_bounds_cpu',
        'test_eyelike_populate_off_main_diagonal_cpu',
        'test_eyelike_with_dtype_cpu',
        'test_eyelike_without_dtype_cpu',
        'test_maxunpool_export_with_output_shape_cpu',
        'test_maxunpool_export_without_output_shape_cpu',
        'test_nonzero_example_cpu',
        'test_onehot_with_axis_cpu',
        'test_onehot_without_axis_cpu',
        'test_scan9_sum_cpu',
        'test_scan_sum_cpu',
        'test_shrink_hard_cpu',
        'test_shrink_soft_cpu',
        'test_sinh_cpu',
        'test_sinh_example_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_only_bigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu',
        'test_basic_convinteger_cpu',
        'test_cast_FLOAT_to_STRING_cpu',
        'test_cast_STRING_to_FLOAT_cpu',
        'test_cast_DOUBLE_to_FLOAT16_cpu',
        'test_cast_FLOAT16_to_DOUBLE_cpu',
        'test_cast_FLOAT16_to_FLOAT_cpu',
        'test_cast_FLOAT_to_FLOAT16_cpu',
        'test_matmulinteger_cpu',
        'test_prelu_broadcast_cpu',
        'test_convinteger_with_padding_cpu',
        'test_convtranspose_3d_cpu',
        'test_dequantizelinear_cpu',
        'test_instancenorm_epsilon_cpu',
        'test_instancenorm_example_cpu',
        'test_isinf_cpu',
        'test_isinf_negative_cpu',
        'test_isinf_positive_cpu',
        'test_nonmaxsuppression_center_point_box_format_cpu',
        'test_nonmaxsuppression_flipped_coordinates_cpu',
        'test_nonmaxsuppression_identical_boxes_cpu',
        'test_nonmaxsuppression_limit_output_size_cpu',
        'test_nonmaxsuppression_single_box_cpu',
        'test_nonmaxsuppression_suppress_by_IOU_and_scores_cpu',
        'test_nonmaxsuppression_suppress_by_IOU_cpu',
        'test_nonmaxsuppression_two_batches_cpu',
        'test_nonmaxsuppression_two_classes_cpu',
        'test_prelu_example_cpu',
        'test_qlinearconv_cpu',
        'test_qlinearmatmul_2D_cpu',
        'test_qlinearmatmul_3D_cpu',
        'test_quantizelinear_cpu',
        # Resize test fails due to unknown Scaling factor
        # Requires custom layer/function to bypass
        'test_resize_downsample_linear_cpu',
        'test_resize_downsample_nearest_cpu',
        'test_resize_nearest_cpu',
        'test_resize_upsample_linear_cpu',
        'test_resize_upsample_nearest_cpu',
        'test_reversesequence_time_cpu',
        'test_slice_cpu',
        'test_slice_default_axes_cpu',
        'test_slice_default_steps_cpu',
        'test_slice_end_out_of_bounds_cpu',
        'test_slice_neg_cpu',
        'test_slice_neg_steps_cpu',
        'test_strnormalizer_export_monday_casesensintive_lower_cpu',
        'test_strnormalizer_export_monday_casesensintive_nochangecase_cpu',
        'test_strnormalizer_export_monday_casesensintive_upper_cpu',
        'test_strnormalizer_export_monday_empty_output_cpu',
        'test_strnormalizer_export_monday_insensintive_upper_twodim_cpu',
        'test_strnormalizer_nostopwords_nochangecase_cpu',
        'test_PReLU_1d_cpu',
        'test_PReLU_2d_cpu',
        'test_strnorm_model_monday_casesensintive_lower_cpu',
        'test_strnorm_model_monday_casesensintive_nochangecase_cpu',
        'test_strnorm_model_monday_casesensintive_upper_cpu',
        'test_strnorm_model_monday_empty_output_cpu',
        'test_strnorm_model_monday_insensintive_upper_twodim_cpu',
        'test_strnorm_model_nostopwords_nochangecase_cpu',
        'test_averagepool_2d_ceil_cpu',
        'test_convtranspose_pad_cpu',
        'test_maxpool_2d_ceil_cpu',
        'test_maxpool_2d_dilations_cpu',

        # Incorrect test setup
        'test_mod_float_mixed_sign_example_cpu',
        'test_mod_int64_mixed_sign_example_cpu',
        
        # Need dead code elimination and removing unused inputs
        'test_batchnorm_epsilon_cpu',
        'test_batchnorm_example_cpu',

        # recurrent tests.
        'test_operator_rnn_cpu',
        'test_operator_rnn_single_layer_cpu',
        'test_operator_lstm_cpu',
        'test_gru_defaults_cpu',
        'test_gru_with_initial_bias_cpu',
        'test_lstm_defaults_cpu',
        'test_lstm_with_initial_bias_cpu',
        'test_lstm_with_peepholes_cpu',
        'test_simple_rnn_defaults_cpu',
        'test_simple_rnn_with_initial_bias_cpu',

        # exclude all the model zoo tests. They are tested elsewhere.
        'test_bvlc_alexnet',
        'test_resnet50',
        'test_vgg16',
        'test_vgg19',
        'test_densenet121',
        'test_inception_v1',
        'test_inception_v2',
        'test_shufflenet',
        'test_squeezenet',
        'test_zfnet',
        'test_maxunpool_export_with_output_shape_cpu',
        'test_maxunpool_export_without_output_shape_cpu',
        'test_nonzero_example_cpu',
        'test_onehot_with_axis_cpu',
        'test_onehot_without_axis_cpu',
        'test_scan9_sum_cpu',
        'test_scan_sum_cpu',
        'test_shrink_hard_cpu',
        'test_shrink_soft_cpu',
        'test_sinh_cpu',
        'test_sinh_example_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_batch_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_batch_uniandbigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_only_bigrams_skip0_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_levelempty_cpu',
        'test_tfidfvectorizer_tf_onlybigrams_skip5_cpu',
        'test_tfidfvectorizer_tf_uniandbigrams_skip5_cpu',
    ]

    if SupportedVersion.is_nd_array_supported(MINIMUM_IOS_DEPLOYMENT_TARGET):
        for each in unsupported_tests_coreml3:
            backend_test.exclude(each)
    else:
        for each in unsupported_tests_till_coreml2:
            backend_test.exclude(each)
    
# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(CoreMLTestingBackend, __name__)

# exclude unsupported test cases
exclude_test_cases(backend_test)

globals().update(backend_test
                 .enable_report()
                 .test_cases)


if __name__ == '__main__':
    unittest.main()

