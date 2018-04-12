from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Dict, Text, Any, Callable
from coremltools.models.neural_network import NeuralNetworkBuilder  #type: ignore
from ._graph import Node

class ErrorHandling(object):
  '''
  To handle errors and addition of custom layers
  '''

  def __init__(self,
               add_custom_layers = False, # type: bool
               add_custom_layers_for_unsupported_attributes = False, # type: bool
               custom_conversion_functions = dict(), # type: Dict[Text, Any]
               ):
      # type: (...) -> None
      self.add_custom_layers = add_custom_layers
      self.add_custom_layers_for_unsupported_attributes = add_custom_layers_for_unsupported_attributes
      self.custom_conversion_functions = custom_conversion_functions


  def unsupported_op(self,
                     builder, # type : NeuralNetworkBuilder
                     node,  # type: Node
                    ):
      # type: (...) -> Callable[[Any, Node], None]
      '''
      Either raise an error for an unsupported op type or add a custom layer.
      '''
      raise TypeError(
        "ONNX node of type {} is not supported.\n".format(node.op_type,)
      )


  def unsupported_op_configuration(self,
                                   builder, # type: NeuralNetworkBuilder
                                   node, # type: Node
                                   err_message, # type: Text
                                   ):
      # type: (...) -> None
      '''
      Either raise an error for an unsupported attribute or add a custom layer.
      '''
      raise TypeError(
        "Error while converting op of type: {}. Error message: {}\n".format(node.op_type, err_message, )
      )


  def missing_initializer(self,
                          node, # type: Node
                          err_message, # type: Text
                          ):
      # type: (...) -> None
      '''
      Either raise an error for an unsupported attribute or add a custom layer.
      '''
      raise ValueError(
        "Missing initializer error in op of type {}, with input name = {}, "
        "output name = {}. Error message: {}\n".
        format(node.op_type, node.inputs[0], node.outputs[0], err_message)
      )



