from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import numpy_helper


def _input_from_onnx_input(input):
    name = input.name
    type = input.type.tensor_type.elem_type
    shape = tuple([d.dim_value for d in input.type.tensor_type.shape.dim])
    return (name, type, shape)


def _convertAttributeProto(onnx_arg):
    """
    Convert an ONNX AttributeProto into an appropriate Python object
    for the type.
    NB: Tensor attribute gets returned as numpy array
    """
    if onnx_arg.HasField('f'):
        return onnx_arg.f
    elif onnx_arg.HasField('i'):
        return onnx_arg.i
    elif onnx_arg.HasField('s'):
        return onnx_arg.s
    elif onnx_arg.HasField('t'):
        return numpy_helper.to_array(onnx_arg.t)
    elif len(onnx_arg.floats):
        return list(onnx_arg.floats)
    elif len(onnx_arg.ints):
        return list(onnx_arg.ints)
    elif len(onnx_arg.strings):
        return list(onnx_arg.strings)
    else:
        raise ValueError("Unsupported ONNX attribute: {}".format(onnx_arg))


class Attributes(dict):
    @staticmethod
    def from_onnx(args):
        d = Attributes()
        for arg in args:
            d[arg.name] = _convertAttributeProto(arg)
        return d


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
        attrs = Attributes.from_onnx(node.attribute)
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

    def transformed(self, transformers):
        graph = self
        for transformer in transformers:
            graph = transformer(graph)
        return graph

    def has_edge_name(self, name):
        '''
        Check if name is already used for graph inputs/outputs or for nodes
        inputs/outputs
        '''
        names = set()
        for input in self.inputs:
            names.add(input[0])
        for output in self.outputs:
            names.add(output[0])
        for node in self.nodes:
            names.update(node.inputs)
            names.update(node.outputs)
        return name in names

    def get_unique_edge_name(self, name):
        n_ = name
        i = 0
        while self.has_edge_name(n_):
            n_ = "{}_{}".format(name, i)
            i += 1
        return n_

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
                inputs.append(_input_from_onnx_input(i))

        outputs = []
        for o in graph.output:
            outputs.append(_input_from_onnx_input(o))

        for node_ in nodes_:
            for input_ in node_.inputs:
                if input_ in nodes_by_output:
                    node_.parents.append(nodes_by_output[input_])
            for output_ in node_.outputs:
                if output_ in nodes_by_input:
                    node_.children.extend(nodes_by_input[output_])

        return Graph(nodes_, inputs, outputs)
