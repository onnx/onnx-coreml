from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import onnx.backend.test

from onnx_coreml.backend import CoreMLBackend


# import all test cases at global scope to make them visible to python.unittest
test_cases = onnx.backend.test.BackendTest(CoreMLBackend, __name__).test_cases
# TODO: support node tests
del test_cases['OnnxBackendNodeTest']
globals().update(
    test_cases
)


if __name__ == '__main__':
    unittest.main()
