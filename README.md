# Convert ONNX models into Apple Core ML format.

`onnx-coreml` package is **no longer supported** and will **not be maintained**.

## PyTorch Models

For converting PyTorch models to CoreML format, the recommended approach is to use **new** PyTorch to Core ML converter, introduced in the [`coremltools 4.0`](https://github.com/apple/coremltools) python package.
Please read the  coremltools documentation on [PyTorch conversion](https://coremltools.readme.io/docs/pytorch-conversion) for example usage.

## ONNX Models

Code for ONNX to Core ML conversion is now available through `coremltools` python package and `coremltools.converters.onnx.convert` is the only supported API for conversion. To read more about exporting ONNX models to Core ML format, please visit coremltools documentation on [ONNX conversion.](https://coremltools.readme.io/docs/onnx-conversion)

Note: ONNX converter is not under any active feature development. For access to bug fixes, community support and requests, please use [coremltools](https://github.com/apple/coremltools) github repository. 

## Installation 
To install coremltools package, please follow [these instructions](https://coremltools.readme.io/docs/installation) in the coremltools documentation.

## License
Copyright © 2018 by Apple Inc., Facebook Inc., and Prisma Labs Inc.

Use of this source code is governed by the [MIT License](https://opensource.org/licenses/MIT) that can be found in the LICENSE.txt file.
