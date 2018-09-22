from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import unittest
import onnx
from onnx_coreml import convert
import numpy as np
import numpy.testing as npt  # type: ignore
from tests._test_utils import _assert_outputs
import torch # type: ignore
import torch.nn as nn # type: ignore
import shutil
import tempfile
import os

def _test_torch_model_single_io(torch_model, torch_input_shape, coreml_input_shape):
    # run torch model
    torch_input = torch.rand(*torch_input_shape)
    torch_out = torch_model(torch_input).detach().numpy()

    # convert to onnx model
    model_dir = tempfile.mkdtemp()
    onnx_file = os.path.join(model_dir, 'torch_model.onnx')
    torch.onnx.export(torch_model, torch_input, onnx_file)
    onnx_model = onnx.load(onnx_file)

    # convert to coreml and run
    coreml_model = convert(onnx_model)
    output_name = [o.name for o in onnx_model.graph.output][0]
    initializer_names = {t.name for t in onnx_model.graph.initializer}
    input_name = [i.name for i in onnx_model.graph.input if i.name not in initializer_names][0]
    input_numpy = torch_input.detach().numpy()
    input_dict = {input_name: np.reshape(input_numpy, coreml_input_shape)} # type: ignore
    coreml_out = coreml_model.predict(input_dict, useCPUOnly=True)[output_name]

    # compare
    _assert_outputs([torch_out], [coreml_out], decimal=4) # type: ignore

    # delete onnx model
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)

class OnnxModelTest(unittest.TestCase):

    def test_linear_no_bias(self):  # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.simple_nn = nn.Sequential(nn.Linear(256, 128, bias=False), nn.ReLU())

            def forward(self, x):
                return self.simple_nn(x)

        torch_model = Net() # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1,256), (256)) # type: ignore

    def test_dynamic_reshape(self):  # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.conv = nn.Conv2d(in_channels=3,
                                      out_channels=32,
                                      kernel_size=(3, 3),
                                      stride=1, padding=0,
                                      bias=True)

            def forward(self, x):
                x = self.conv(x)
                x = x.view(x.size()[0], -1)
                return x

        torch_model = Net() # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 3, 100, 100), (3, 100, 100)) # type: ignore

    def test_const_initializer1(self):  # typr: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.ones = torch.nn.Parameter(torch.ones(1,))

            def forward(self, x):
                y = x + self.ones
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 3), (3,))  # type: ignore


    def test_const_initializer2(self):  # typr: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                y = x + torch.nn.Parameter(torch.ones(2, 3))
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 2, 3), (1, 2, 3))  # type: ignore



if __name__ == '__main__':
    unittest.main()

