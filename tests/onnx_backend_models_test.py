from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx

import onnx.backend.test

from onnx_coreml._backend import CoreMLBackend, CoreMLBackendND

from coremltools.models.utils import macos_version

# Disable Rank 5 mapping for ONNX backend testing
DISABLE_RANK5_MAPPING = False

MIN_MACOS_VERSION_10_15 = (10, 15)
# If MACOS version is less than 10.15
# Then force testing on CoreML 2.0
if macos_version() < MIN_MACOS_VERSION_10_15:
    DISABLE_RANK5_MAPPING = False

# import all test cases at global scope to make them visible to python.unittest
backend_test = onnx.backend.test.BackendTest(CoreMLBackendND if DISABLE_RANK5_MAPPING else CoreMLBackend, __name__)

# Only include the big models tests
backend_test.include('test_resnet50')
backend_test.include('test_inception_v1')
backend_test.include('test_inception_v2')
backend_test.include('test_densenet121')
backend_test.include('test_shufflenet')
backend_test.include('test_squeezenet')
backend_test.include('test_bvlc_alexnet')
backend_test.include('test_zfnet512')
backend_test.include('test_vgg19')

globals().update(backend_test
                 .enable_report()
                 .test_cases)


if __name__ == '__main__':
    unittest.main()
