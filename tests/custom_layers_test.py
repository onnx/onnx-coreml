from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tests._test_utils import _onnx_create_model
from onnx import helper, numpy_helper, ModelProto, TensorProto
from onnx_coreml import convert
from coremltools.proto import NeuralNetwork_pb2 #type: ignore

def _make_model_clip_exp_topk(): # type: (...) -> ModelProto
  '''
  make a very simple model for testing: input->clip->exp->topk->2 outputs
  '''
  inputs = [('input0', (10,))]
  outputs = [('output_values', (3,), TensorProto.FLOAT), ('output_indices', (3,), TensorProto.INT64)]
  acos = helper.make_node("Acos",
                          inputs=[inputs[0][0]],
                          outputs=['acos_out'])
  exp = helper.make_node("Exp",
                        inputs=[acos.output[0]],
                        outputs=['exp_out'])
  topk = helper.make_node("TopK",
                        inputs=[exp.output[0]],
                        outputs=[outputs[0][0], outputs[1][0]],
                        axis=0, k=3)
  return _onnx_create_model([acos, exp, topk], inputs, outputs)

def _make_model_concat_axis3(): # type: (...) -> ModelProto
  '''
  make a simple model: 4-D input1, 4-D input2 -> concat (axis=3)-> output
  '''
  inputs = [('input0', (1,3,10,20)), ('input1', (1,3,10,15))]
  outputs = [('output', (1,3,10,35), TensorProto.FLOAT)]
  concat = helper.make_node("Concat",
                            inputs=[inputs[0][0], inputs[1][0]],
                            outputs=[outputs[0][0]],
                            axis=3)
  return _onnx_create_model([concat], inputs, outputs)


class CustomLayerTest(unittest.TestCase):

  def test_unsupported_ops(self):  # type: () -> None

    onnx_model = _make_model_clip_exp_topk()
    coreml_model = convert(onnx_model, add_custom_layers=True)

    spec = coreml_model.get_spec()
    layers = spec.neuralNetwork.layers
    self.assertIsNotNone(layers[0].custom)
    self.assertIsNotNone(layers[2].custom)
    self.assertEqual('Acos', layers[0].custom.className)
    self.assertEqual('TopK', layers[2].custom.className)


  def test_unsupported_ops_provide_functions(self):  # type: () -> None

    def convert_acos(node):
      params = NeuralNetwork_pb2.CustomLayerParams()
      params.className = node.op_type
      params.description = "Custom layer that corresponds to the ONNX op {}".format(node.op_type, )
      return params


    def convert_topk(node):
      params = NeuralNetwork_pb2.CustomLayerParams()
      params.className = node.op_type
      params.description = "Custom layer that corresponds to the ONNX op {}".format(node.op_type, )
      params.parameters["axis"].intValue = node.attrs.get('axis', -1)
      params.parameters["k"].intValue = node.attrs['k']
      return params

    onnx_model = _make_model_clip_exp_topk()
    coreml_model = convert(model=onnx_model,
                           add_custom_layers=True,
                           custom_conversion_functions={'Acos':convert_acos, 'TopK':convert_topk})

    spec = coreml_model.get_spec()
    layers = spec.neuralNetwork.layers
    self.assertIsNotNone(layers[0].custom)
    self.assertIsNotNone(layers[2].custom)
    self.assertEqual('Acos', layers[0].custom.className)
    self.assertEqual('TopK', layers[2].custom.className)
    self.assertEqual(0, layers[2].custom.parameters['axis'].intValue)
    self.assertEqual(3, layers[2].custom.parameters['k'].intValue)



  def test_unsupported_op_attribute(self):  # type: () -> None

    onnx_model = _make_model_concat_axis3()
    coreml_model = convert(onnx_model, add_custom_layers=True)

    spec = coreml_model.get_spec()
    layers = spec.neuralNetwork.layers
    self.assertIsNotNone(layers[0].custom)
    self.assertEqual('Concat', layers[0].custom.className)

  def test_unsupported_op_attribute_provide_functions(self):  # type: () -> None

    def convert_concat(node):
      params = NeuralNetwork_pb2.CustomLayerParams()
      params.className = node.op_type
      params.description = "Custom layer that corresponds to the ONNX op {}".format(node.op_type, )
      params.parameters["axis"].intValue = node.attrs['axis']
      return params

    onnx_model = _make_model_concat_axis3()
    coreml_model = convert(onnx_model, add_custom_layers=True,
                           custom_conversion_functions={'Concat': convert_concat})

    spec = coreml_model.get_spec()
    layers = spec.neuralNetwork.layers
    self.assertIsNotNone(layers[0].custom)
    self.assertEqual('Concat', layers[0].custom.className)
    self.assertEqual(3, layers[0].custom.parameters['axis'].intValue)
