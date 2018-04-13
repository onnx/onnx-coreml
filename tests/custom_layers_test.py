from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

from tests._test_utils import _onnx_create_model
from onnx import helper, numpy_helper
from onnx_coreml import convert
from coremltools.proto import NeuralNetwork_pb2 #type: ignore

def _make_model_clip_exp_topk():
  '''
  make a very simple model for testing 
  '''
  inputs = [('input0', (10,))]
  outputs = [('output_values', (3,)), ('output_indices', (3,))]
  clip = helper.make_node("Clip",
                          inputs=[inputs[0][0]],
                          outputs=['clip_out'],
                          max=10.0, min=5.0)
  exp = helper.make_node("Exp",
                        inputs=[clip.output[0]],
                        outputs=['exp_out'])
  topk = helper.make_node("TopK",
                        inputs=[exp.output[0]],
                        outputs=[outputs[0][0], outputs[1][0]],
                        axis=0, k=3)
  return _onnx_create_model([clip, exp, topk], inputs, outputs)


class CustomLayerTest(unittest.TestCase):

  def test_unsupported_ops_no_custom_functions(self):  # type: () -> None

    onnx_model = _make_model_clip_exp_topk()
    coreml_model = convert(onnx_model, add_custom_layers=True)

    spec = coreml_model.get_spec()
    layers = spec.neuralNetwork.layers
    self.assertIsNotNone(layers[0].custom)
    self.assertIsNotNone(layers[2].custom)
    self.assertEqual('Clip', layers[0].custom.className)
    self.assertEqual('TopK', layers[2].custom.className)


  def test_unsupported_ops(self):  # type: () -> None

    def convert_clip(node):
      params = NeuralNetwork_pb2.CustomLayerParams()
      params.className = node.op_type
      params.description = "Custom layer that corresponds to the ONNX op {}".format(node.op_type, )
      params.parameters["min"].doubleValue = node.attrs['min']
      params.parameters["max"].doubleValue = node.attrs['max']
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
                           custom_conversion_functions={'Clip':convert_clip, 'TopK':convert_topk})

    spec = coreml_model.get_spec()
    layers = spec.neuralNetwork.layers
    self.assertIsNotNone(layers[0].custom)
    self.assertIsNotNone(layers[2].custom)
    self.assertEqual('Clip', layers[0].custom.className)
    self.assertEqual('TopK', layers[2].custom.className)
    self.assertEqual(10.0, layers[0].custom.parameters['max'].doubleValue)
    self.assertEqual(5.0, layers[0].custom.parameters['min'].doubleValue)
    self.assertEqual(0, layers[2].custom.parameters['axis'].intValue)
    self.assertEqual(3, layers[2].custom.parameters['k'].intValue)


  # def test_unsupported_op_attribute(self):  # type: () -> None
  #
  #
  # def test_unsupported_op_attribute_no_custom_function(self):  # type: () -> None


    # with open('/tmp/node_model.onnx', 'wb') as f:
    #   s = onnx_model.SerializeToString()
    #   f.write(s)
    # coreml_model.save('/tmp/model.mlmodel')
    # print(spec)