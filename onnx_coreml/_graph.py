from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import numpy_helper


class Node(object):
    def __init__(self, name, op_type, attrs, inputs, outputs):
        self.name = name
        self.op_type = op_type
        self.attrs = attrs
        self.inputs = inputs
        self.outputs = outputs
        self.input_tensors = {}
        self.parents = []
        self.children = []
        self.metadata = {}

    def add_parent(self, parent_node):
        assert parent_node not in self.parents
        self.parents.append(parent_node)
        if self not in parent_node.children:
            parent_node.children.append(self)

    def add_child(self, child_node):
        assert child_node not in self.children
        self.children.append(child_node)
        if self not in child_node.parents:
            child_node.parents.append(self)

    def get_only_parent(self):
        if len(self.parents) != 1:
            raise ValueError('Node ({}) expected to have 1 parent. Found {}.'
                             .format(self, len(self.parents)))
        return self.parents[0]

    @staticmethod
    def from_onnx(node):
        attrs = {attr.name: attr for attr in node.attribute}
        name = str(node.name)
        if len(name) == 0:
            name = "_".join(node.output)
        return Node(
            name, node.op_type, attrs, list(node.input), list(node.output)
        )


class Graph(object):
    def __init__(self, nodes, inputs, outputs):
        self.nodes = nodes
        self.inputs = inputs
        self.outputs = outputs

    @staticmethod
    def _input_from_onnx_input(input):
        name = input.name
        type = input.type.tensor_type.elem_type
        shape = tuple([d.dim_value for d in input.type.tensor_type.shape.dim])
        return (name, type, shape)

    def transformed(self, transformers):
        graph = self
        for transformer in transformers:
            graph = transformer(graph)
        return graph

    @staticmethod
    def from_onnx(graph):
        input_tensors = {
            t.name: numpy_helper.to_array(t) for t in graph.initializer
        }
        nodes_ = []
        nodes_by_input = {}
        nodes_by_output = {}
        for node in graph.node:
            node_ = Node.from_onnx(node)
            for input_ in node_.inputs:
                if input_ in input_tensors:
                    node_.input_tensors[input_] = input_tensors[input_]
                else:
                    if input_ in nodes_by_input:
                        input_nodes = nodes_by_input[input_]
                    else:
                        input_nodes = []
                        nodes_by_input[input_] = input_nodes
                    input_nodes.append(node_)
            for output_ in node_.outputs:
                nodes_by_output[output_] = node_
            nodes_.append(node_)

        inputs = []
        for i in graph.input:
            if i.name not in input_tensors:
                inputs.append(Graph._input_from_onnx_input(i))

        outputs = []
        for o in graph.output:
            outputs.append(Graph._input_from_onnx_input(o))

        for node_ in nodes_:
            for input_ in node_.inputs:
                if input_ in nodes_by_output:
                    node_.parents.append(nodes_by_output[input_])
            for output_ in node_.outputs:
                if output_ in nodes_by_input:
                    node_.children.extend(nodes_by_input[output_])

        return Graph(nodes_, inputs, outputs)
