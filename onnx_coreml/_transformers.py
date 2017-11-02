from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from ._graph import Graph


class SubNodeFuser(object):
    '''
    An abstract helper for merging a children with its parent.
    '''
    def __call__(self, graph):
        nodes = graph.nodes
        fused_nodes = []
        for node in nodes:
            if len(node.parents) != 1:
                # We're only fusing nodes with single parents
                continue
            parent = node.get_only_parent()
            if len(parent.children) != 1:
                # We can only fuse a node if its parent's
                # value isn't used by any other node.
                continue
            if not self.is_eligible_pair(parent, node):
                continue
            # Rewrite the fused node's children to its parent.
            for child in node.children:
                child.parents.remove(node)
                parent.add_child(child)
            # Disconnect the fused node from the graph.
            parent.children.remove(node)
            parent.outputs = node.outputs
            fused_nodes.append(node)
            # Let the sub-class merge the fused node in any arbitrary way.
            self.merge(parent, node)
        transformed_nodes = [node for node in nodes if node not in fused_nodes]
        return Graph(transformed_nodes, graph.inputs, graph.outputs)

    def is_eligible_pair(self, parent, child):
        '''Returns true if this parent/child pair is eligible for fusion.'''
        raise NotImplementedError('Must be implemented by subclass.')

    def merge(self, parent, child):
        '''Merge the child node into the parent.'''
        raise NotImplementedError('Must be implemented by subclass')


class ConvAddFuser(SubNodeFuser):
    '''
    Fuses Add layer into parent convolution layer.
    '''
    def is_eligible_pair(self, parent, child):
        if parent.op_type != 'Conv':
            return False
        if child.op_type != 'Add':
            return False
        if 'broadcast' not in child.attrs:
            return False
        if 'axis' not in child.attrs:
            return False

        broadcast = child.attrs['broadcast']
        if broadcast != 1:
            return False

        axis = child.attrs['axis']
        if axis != 1:
            return False

        return True

    def merge(self, parent, child):
        output_channels = parent.input_tensors[parent.inputs[1]].shape[0]
        if len(parent.inputs) > 2:
            bias_input_name = parent.inputs[2]
            bias = parent.input_tensors[bias_input_name]
        else:
            bias_input_name = "{}_bias".format(parent.name,)
            parent.inputs.append(bias_input_name)
            bias = np.zeros(
                (output_channels,), dtype=np.float32
            )
            parent.input_tensors[bias_input_name] = bias
        bias = bias + child.input_tensors[child.inputs[1]]
        parent.input_tensors[bias_input_name] = bias


class BNBroadcastedMulFuser(SubNodeFuser):
    '''
    Fuses Mul into BatchNorm
    '''
    def is_eligible_pair(self, parent, child):
        if parent.op_type != 'BatchNormalization':
            return False
        if child.op_type != 'Mul':
            return False
        if "broadcast" not in child.attrs:
            return False
        if child.attrs["broadcast"] != 1:
            return False
        if "axis" not in child.attrs:
            return False
        if child.attrs["axis"] != 1:
            return False
        if child.inputs[1] not in child.input_tensors:
            return False
        return True

    def merge(self, parent, child):
        weight = parent.input_tensors[parent.inputs[1]]
        bias = parent.input_tensors[parent.inputs[2]]
        W = child.input_tensors[child.inputs[1]]
        parent.input_tensors[parent.inputs[1]] = np.multiply(weight, W)
        parent.input_tensors[parent.inputs[2]] = np.multiply(bias, W)


class BNBroadcastedAddFuser(SubNodeFuser):
    '''
    Fuses Add into BatchNorm
    '''
    def is_eligible_pair(self, parent, child):
        if parent.op_type != 'BatchNormalization':
            return False
        if child.op_type != 'Add':
            return False
        if "broadcast" not in child.attrs:
            return False
        if child.attrs["broadcast"] != 1:
            return False
        if "axis" not in child.attrs:
            return False
        if child.attrs["axis"] != 1:
            return False
        if len(child.inputs) != 2:
            return False
        if child.inputs[1] not in child.input_tensors:
            return False
        return True

    def merge(self, parent, child):
        bias = parent.input_tensors[parent.inputs[2]]
        b = child.input_tensors[child.inputs[1]]
        parent.input_tensors[parent.inputs[2]] = bias + b


class DropoutRemover(SubNodeFuser):
    '''
    Removes Dropout layer
    '''
    def is_eligible_pair(self, parent, child):
        return child.op_type == "Dropout"

    def merge(self, parent, child):
        pass


class ReshapeInitTensorFuser(object):
    '''
    Fuses Reshape operator if it is used only to reshape blob in
    graph initializer. We can reshape here instead of runtime.
    '''

    def __call__(self, graph):
        nodes = graph.nodes
        removed = []
        for node in nodes:
            if node.op_type != 'Reshape':
                continue
            if len(node.input_tensors) != 1:
                continue
            tensor_name = node.input_tensors.keys()[0]
            if tensor_name != node.inputs[0]:
                continue
            assert len(node.parents) == 0

            removed.append(node)
            output_name = node.outputs[0]

            tensor = node.input_tensors[tensor_name]
            shape = tuple(node.attrs["shape"])
            reshaped_tensor = tensor.reshape(shape)

            for child in node.children:
                child.parents.remove(node)
                child.input_tensors[output_name] = reshaped_tensor

        transformed_nodes = [node for node in nodes if node not in removed]
        return Graph(transformed_nodes, graph.inputs, graph.outputs)


class DanglingOutputsRemover(object):
    '''
    Removes unused outputs
    '''

    def __call__(self, graph):
        nodes = graph.nodes
        graph_output_names = set([o[0] for o in graph.outputs])
        for node in nodes:
            removed_outputs = set()
            for output in node.outputs:
                if output in graph_output_names:
                    continue
                children_inputs = set()
                for child in node.children:
                    for input_ in child.inputs:
                        children_inputs.add(input_)
                if output in children_inputs:
                    continue
                removed_outputs.add(output)
            node.outputs = [out for out in node.outputs
                            if out not in removed_outputs]
        return graph


class OutputRenamer(object):
    '''
    Rename outputs according to mapping
    '''
    def __init__(self, mapping):
        self.mapping = mapping

    def __call__(self, graph):
        mapping = self.mapping.copy()
        nodes = graph.nodes
        for node in nodes:
            for i in range(len(node.outputs)):
                output = node.outputs[i]
                if output not in mapping:
                    continue
                node.outputs[i] = mapping[output]
                for child in node.children:
                    for j in range(len(child.inputs)):
                        input_ = child.inputs[j]
                        if input_ != output:
                            continue
                        child.inputs[j] = mapping[output]
                del mapping[output]
                if len(mapping) == 0:
                    break
        return graph
