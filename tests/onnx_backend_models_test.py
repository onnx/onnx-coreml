from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx

import onnx.backend.test

from onnx_coreml._backend import CoreMLBackend

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(CoreMLBackend, __name__)

# Only include the big models tests
backend_test.include('test_resnet50')
backend_test.include('test_inception_v1')
backend_test.include('test_inception_v2')
backend_test.include('test_densenet121')
backend_test.include('test_shufflenet')
backend_test.include('test_squeezenet')

#Slow tests. Skipping for now.
backend_test.exclude('test_vgg19')
backend_test.exclude('test_bvlc_alexnet')
backend_test.exclude('test_zfnet')

globals().update(backend_test
                 .enable_report()
                 .test_cases)


if __name__ == '__main__':
    unittest.main()
