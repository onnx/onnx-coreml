from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest
import numpy as np

from onnx import helper, numpy_helper

from onnx_coreml._graph import Graph
from onnx_coreml._transformers import ConvAddFuser
from tests._test_utils import _onnx_create_model, \
    _conv_pool_output_size, _random_array


class ConvAddFuserTest(unittest.TestCase):
    def test_fuse_conv_without_bias(self):
        kernel_shape = (3, 2)
        strides = (2, 3)
        pads = (4, 2, 4, 2)
        dilations = (1, 2)
        group = 1
        weight = numpy_helper.from_array(
            _random_array((16, 3, 3, 2)), name="weight"
        )

        input_shape = (1, 3, 224, 224)
        output_size = _conv_pool_output_size(input_shape, dilations,
                                             kernel_shape, pads, strides)

        output_shape = (1, int(weight.dims[0]), output_size[0], output_size[1])

        inputs = [('input0', input_shape)]
        outputs = [('output0', output_shape)]

        conv = helper.make_node(
            "Conv",
            inputs=[inputs[0][0], "weight"],
            outputs=["conv_output"],
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides
        )

        b = _random_array((int(weight.dims[0]),))
        bias = numpy_helper.from_array(
            b, name="bias"
        )

        add = helper.make_node(
            "Add",
            inputs=[conv.output[0], "bias"],
            outputs=[outputs[0][0]],
            broadcast=1,
            axis=1
        )

        model = _onnx_create_model(
            [conv, add], inputs, outputs, [weight, bias]
        )
        graph_ = Graph.from_onnx(model.graph)
        fused_graph = graph_.transformed([ConvAddFuser()])

        self.assertEqual(len(fused_graph.nodes), 1)
        node = fused_graph.nodes[0]
        self.assertEqual(len(node.inputs), 3)
        np.testing.assert_equal(node.input_tensors[node.inputs[2]], b)
        self.assertEqual(fused_graph.nodes[0].outputs[0], outputs[0][0])

    def test_fuse_conv_with_bias(self):
        kernel_shape = (3, 2)
        strides = (2, 3)
        pads = (4, 2, 4, 2)
        dilations = (1, 2)
        group = 1
        weight = numpy_helper.from_array(
            _random_array((16, 3, 3, 2)), name="weight"
        )
        b = _random_array((int(weight.dims[0]),))
        bias = numpy_helper.from_array(
            b, name="bias"
        )

        input_shape = (1, 3, 224, 224)
        output_size = _conv_pool_output_size(input_shape, dilations,
                                             kernel_shape, pads, strides)

        output_shape = (1, int(weight.dims[0]), output_size[0], output_size[1])

        inputs = [('input0', input_shape)]
        outputs = [('output0', output_shape)]

        conv = helper.make_node(
            "Conv",
            inputs=[inputs[0][0], "weight", "bias"],
            outputs=["conv_output"],
            dilations=dilations,
            group=group,
            kernel_shape=kernel_shape,
            pads=pads,
            strides=strides
        )

        add = helper.make_node(
            "Add",
            inputs=[conv.output[0], "bias"],
            outputs=[outputs[0][0]],
            broadcast=1,
            axis=1
        )

        model = _onnx_create_model(
            [conv, add], inputs, outputs, [weight, bias]
        )
        graph_ = Graph.from_onnx(model.graph)
        fused_graph = graph_.transformed([ConvAddFuser()])

        self.assertEqual(len(fused_graph.nodes), 1)
        node = fused_graph.nodes[0]
        self.assertEqual(len(node.inputs), 3)
        np.testing.assert_equal(node.input_tensors[node.inputs[2]], b * 2)
        self.assertEqual(fused_graph.nodes[0].outputs[0], outputs[0][0])


if __name__ == '__main__':
    unittest.main()
