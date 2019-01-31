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

np.random.seed(10)
torch.manual_seed(10)

DEBUG = False

def _test_torch_model_single_io(torch_model, torch_input_shape, coreml_input_shape):
    # run torch model
    torch_input = torch.rand(*torch_input_shape)
    torch_out_raw = torch_model(torch_input)
    if isinstance(torch_out_raw, tuple):
        torch_out = torch_out_raw[0].detach().numpy()
    else:
        torch_out = torch_out_raw.detach().numpy()

    # convert to onnx model
    model_dir = tempfile.mkdtemp()
    if DEBUG:
        model_dir = '/tmp'
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
    if DEBUG:
        coreml_model.save(model_dir + '/torch_model.mlmodel')
        print('coreml_out')
        print(np.squeeze(coreml_out))
        print('torch_out')
        print(np.squeeze(torch_out))
        print('coreml out shape ', coreml_out.shape)
        print('torch out shape: ', torch_out.shape)

    # compare
    _assert_outputs([torch_out], [coreml_out], decimal=4) # type: ignore

    # delete onnx model
    if not DEBUG:
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

    def test_linear_bias(self):  # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.simple_nn = nn.Sequential(nn.Linear(256, 128, bias=True), nn.ReLU())

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

    def test_const_initializer1(self):  # type: () -> None
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


    def test_const_initializer2(self):  # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                y = x + torch.nn.Parameter(torch.ones(2, 3))
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 2, 3), (1, 2, 3))  # type: ignore

    def test_conv2D_transpose(self): # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.convT = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, output_padding=1, padding=1, groups=1)

            def forward(self, x):
                y = self.convT(x)
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 1, 2, 2), (1, 2, 2))  # type: ignore

    def test_conv2D_transpose_groups(self): # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.convT = torch.nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, output_padding=1, padding=1, groups=2)

            def forward(self, x):
                y = self.convT(x)
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 4, 8, 8), (4, 8, 8))  # type: ignore

    def test_conv2D_transpose_2(self): # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.convT = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=3, output_padding=2, padding=1, groups=1)

            def forward(self, x):
                y = self.convT(x)
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 1, 3, 3), (1, 3, 3))  # type: ignore

    def test_pow(self): # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()

            def forward(self, x):
                y = x.pow(3)
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (3, 2, 3), (3, 2, 3))  # type: ignore

    def test_lstm(self):  # type: () -> None
        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.lstm = nn.LSTM(input_size=256,
                                hidden_size=64,
                                num_layers=1)

            def forward(self, x):
                y = self.lstm(x)
                return y

        torch_model = Net()  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (3, 1, 256), (3, 1, 256))  # type: ignore

    def test_1d_conv(self):
        class Net(nn.Module):
            def __init__(self, in_channels,
                               out_channels,
                               kernel_size,
                               stride=1,
                               dilation=1,
                               groups=1,
                               bias=True):

                super(Net, self).__init__()

                self.conv = torch.nn.Conv1d(in_channels,
                                            out_channels,
                                            kernel_size=kernel_size,
                                            stride=stride,
                                            padding=0,
                                            dilation=dilation,
                                            groups=groups,
                                            bias=bias)

                self.__padding = (kernel_size - 1) * dilation

            def forward(self, x):
                result = self.conv(x)
                if self.__padding != 0:
                    return result[:, :, :-self.__padding]
                return result

        B = 1
        Cin = 5
        Cout = 11
        k = 3
        Win = 15
        torch_model = Net(Cin, Cout, k)  # type: ignore
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, Cin, Win), (Cin, 1, Win))  # type: ignore


if __name__ == '__main__':
    unittest.main()
    #suite = unittest.TestSuite()
    #suite.addTest(OnnxModelTest("test_pow"))
    #unittest.TextTestRunner().run(suite)
