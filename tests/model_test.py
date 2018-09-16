from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import unittest
import onnx
from tests._test_utils import _test_onnx_model

class OnnxModelTest(unittest.TestCase):

    def test_linear_no_bias(self):  # type: () -> None

        import torch
        import torch.nn as nn

        class Net(nn.Module):
            def __init__(self):
                super(Net, self).__init__()
                self.simple_nn = nn.Sequential(nn.Linear(256, 128, bias=False), nn.ReLU())

            def forward(self, x):
                return self.simple_nn(x)

        model = Net()
        model.train(False)
        input = torch.rand(1, 256)
        torch.onnx.export(model, input, "/tmp/test_linear.onnx")
        onnx_model = onnx.load('/tmp/test_linear.onnx')
        _test_onnx_model(onnx_model, decimal=7)
        print('-' * 80)



if __name__ == '__main__':
    unittest.main()

