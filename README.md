# Convert ONNX models into Apple Core ML format.

[![Build Status](https://travis-ci.org/onnx/onnx-coreml.svg?branch=master)](https://travis-ci.org/onnx/onnx-coreml)

This tool converts [ONNX](https://onnx.ai/) models to Apple Core ML format. To convert Core ML models to ONNX, use [ONNXMLTools](https://github.com/onnx/onnxmltools).

There's a comprehensive [Tutorial](https://github.com/onnx/tutorials/tree/master/examples/CoreML/ONNXLive/README.md) showing how to convert PyTorch style transfer models through ONNX to Core ML models and run them in an iOS app. You can find example for PyTorch model conversion [here](https://github.com/onnx/onnx-coreml/tree/master/examples).

## [New] release onnx-coreml converter with Core ML 3

To try out the new onnx-coreml 1.0 converter with Core ML 3 (>= iOS 13, >= macOS 15),
install coremltools 3.0 and coremltools 1.0

```shell
pip install coremltools==3.0
pip install onnx-coreml==1.0
```

Since 1.0 beta 3, the flag `disable_coreml_rank5_mapping` (which was part of beta 2) has been removed and instead replaced by
the generic argument `target_ios` which can be used to target different versions of Core ML/iOS.
The argument `target_ios` takes a string specifying the target deployment iOS version e.g. '11.2', '12' and '13'.
By default, the converter uses the value of '12'.

For example:

```python
from onnx_coreml import convert
ml_model = convert(model='my_model.onnx', target_ios='13')  # to use Core ML 3
```

## Installation

### Install From PyPI

```bash
pip install -U onnx-coreml
```

### Install From Source

To get the latest version of the converter, install from source by cloning the repository along with its submodules and running the install.sh script. That is,

```bash
git clone --recursive https://github.com/onnx/onnx-coreml.git
cd onnx-coreml
./install.sh
```

### Install From Source (for contributors)

To get the latest version of the converter, install from source by cloning the repository along with its submodules and running the install-develop.sh script. That is,

```bash
git clone --recursive https://github.com/onnx/onnx-coreml.git
cd onnx-coreml
./install-develop.sh
```

## Dependencies

* click
* numpy
* coremltools (3.0+)
* onnx (1.5.0+)

## How to Use

To convert models use single function `convert` from onnx_coreml:

```python
from onnx_coreml import convert
```

```python
def convert(model,
            mode=None,
            image_input_names=[],
            preprocessing_args={},
            image_output_names=[],
            deprocessing_args={},
            class_labels=None,
            predicted_feature_name='classLabel',
            add_custom_layers=False,
            custom_conversion_functions={},
            target_ios='12')
```

The function returns a Core ML model instance that can be saved to a `.mlmodel` file, e.g.:

```python
mlmodel = convert(onnx_model)
mlmodel.save('model.mlmodel')
```

Core ML model spec can be obtained from the model instance, which can be used to update model properties such as output names, input names etc. For e.g.:

```python
import coremltools
from coremltools.models import MLModel

spec = mlmodel.get_spec()
new_mlmodel = MLModel(spec)
coremltools.utils.rename_feature(spec, 'old_output_name', 'new_output_name')
coremltools.utils.save_spec(spec, 'model_new_output_name.mlmodel')
```

For more details see coremltools [documentation](https://apple.github.io/coremltools/#).

### Parameters

```
__model__: ONNX model | str
      An ONNX model with parameters loaded in onnx package or path to file
      with models.

__mode__: str ('classifier', 'regressor' or None)
      Mode of the converted coreml model:
      'classifier', a NeuralNetworkClassifier spec will be constructed.
      'regressor', a NeuralNetworkRegressor spec will be constructed.

__image_input_names__: list of strings
      Name of the inputs to be defined as image type. Otherwise, by default all inputs are MultiArray type.

__preprocessing_args__: dict
      Specify preprocessing parameters, that are be applied to all the image inputs specified through the "image_input_names" parameter.
      'is_bgr', 'red_bias', 'green_bias', 'blue_bias', 'gray_bias',
      'image_scale' keys with the same meaning as

https://apple.github.io/coremltools/generated/coremltools.models.neural_network.html#coremltools.models.neural_network.NeuralNetworkBuilder.set_pre_processing_parameters

__image_output_names__: list of strings
      Name of the outputs to be defined as image type. Otherwise, by default all outputs are MultiArray type.

__deprocessing_args__: dict
      Same as 'preprocessing_args' but for the outputs.

__class_labels__: A string or list of strings.
      As a string it represents the name of the file which contains
      the classification labels (one per line).
      As a list of strings it represents a list of categories that map
      the index of the output of a neural network to labels in a classifier.

__predicted_feature_name__: str
      Name of the output feature for the class labels exposed in the Core ML
      model (applies to classifiers only). Defaults to 'classLabel'

__add_custom_layers__: bool
	  If True, then ['custom'](https://developer.apple.com/documentation/coreml/core_ml_api/integrating_custom_layers?language=objc) layers will be added to the model in place of unsupported onnx ops or for the ops
	  that have unsupported attributes.
	  Parameters for these custom layers should be filled manually by editing the mlmodel
	  or the 'custom_conversion_functions' argument can be used to do the same during the process of conversion

__custom_conversion_fuctions__: dict (str: function)
	  Specify custom function to be used for conversion for given op.
        User can override existing conversion function and provide their own custom implementation to convert certain ops.
        Dictionary key must be string specifying ONNX Op name or Op type and value must be a function implementation available in current context.
        Example usage: {'Flatten': custom_flatten_converter, 'Exp': exp_converter}
        `custom_flatten_converter()` and `exp_converter()` will be invoked instead of internal onnx-coreml conversion implementation for these two Ops;
        Hence, User must provide implementation for functions specified in the dictionary. If user provides two separate functions for node name and node type, then custom function tied to node name will be used. As, function tied to node type is more generic than one tied to node name.
        `custom_conversion_functions` option is different than `add_custom_layers`. Both options can be used in conjuction in which case, custom function will be invoked for provided ops and custom layer will be added for ops with no respective conversion function.
        This option gives finer control to user. One use case could be to modify input attributes or certain graph properties before calling
        existing onnx-coreml conversion function. Note that, It is custom conversion function's responsibility to add respective Core ML layer into builder(coreml tools's NeuralNetworkBuilder).
        Examples: https://github.com/onnx/onnx-coreml/blob/master/tests/custom_layers_test.py#L43

__onnx_coreml_input_shape_map__: dict (str: List[int])
    (Optional)
    (only used if `target_ios` version is less than '13')
    A dictionary with keys corresponding to the model input names. Values are a list of integers that specify
    how the shape of the input is mapped to Core ML. Convention used for Core ML shapes is:
    0: Sequence, 1: Batch, 2: channel, 3: height, 4: width.
    For example, an input of rank 2 could be mapped as [3,4] (i.e. H,W) or [1,2] (i.e. B,C) etc.

__target_ios__: str
      Target Deployment iOS version (default: '12')
      Supported values: '11.2', '12', '13'
      Core ML model produced by the converter will be compatible with the iOS version specified in this argument.
      e.g. if `target_ios` = '12', the converter would only utilize Core ML features released till iOS12
      (equivalently macOS 10.14, watchOS 5 etc).
      Release notes:
      * iOS 11 / Core ML 1: https://github.com/apple/coremltools/releases/tag/v0.8
      * iOS 12 / Core ML 2: https://github.com/apple/coremltools/releases/tag/v2.0
      * iOS 13 / Core ML 3: https://github.com/apple/coremltools/releases/tag/v3.0-beta
```

### Returns

```
__model__: A Core ML model.
```

### CLI
Also you can use command-line script for simplicity:
```
convert-onnx-to-coreml [OPTIONS] ONNX_MODEL
```

The command-line script currently doesn't support all options mentioned above. For more advanced use cases, you have to call the python function directly.

## Running Unit Tests

In order to run unit tests, you need `pytest`.

```shell
pip install pytest
pip install pytest-cov
```

To run all unit tests, navigate to the `tests/` folder and run

```shell
pytest
```

To run a specific unit test, for instance the custom layer test, run

```shell
pytest -s custom_layers_test.py::CustomLayerTest::test_unsupported_ops_provide_functions
```

## Currently Supported

### Models
Models from https://github.com/onnx/models that have been tested to work with this converter:

- BVLC Alexnet
- BVLC Caffenet
- BVLC Googlenet
- BVLC reference_rcnn_ilsvrc13
- Densenet
- Emotion-FERPlus
- Inception V1
- Inception V2
- MNIST
- Resnet50
- Shufflenet
- SqueezeNet
- VGG
- ZFNet

### Examples
You can find examples for converting a model through ONNX -> CoreML [here](https://github.com/onnx/onnx-coreml/tree/master/examples)

### Operators
List of [ONNX operators supported in Core ML 2.0 via the converter](https://github.com/onnx/onnx-coreml/blob/4d8b1cc348e2d6a983a6d38bb6921b6b77b47e76/onnx_coreml/_operators.py#L1893)

List of [ONNX operators supported in Core ML 3.0 via the converter](https://github.com/onnx/onnx-coreml/blob/3af826dfb0f17de4310d989acc7d6c5aea42e407/onnx_coreml/_operators_nd.py#L2233)

Some of the operators are partially compatible with Core ML, for example gemm with more than 1 non constant input is not supported in Core ML 2, or scale as an input for upsample layer is not supported in Core ML 3 etc.
For unsupported ops or unsupported attributes within supported ops, Core ML custom layers or custom functions can be used.
See the testing script `tests/custom_layers_test.py` on how to produce Core ML models with custom layers and custom functions.

## License
Copyright © 2018 by Apple Inc., Facebook Inc., and Prisma Labs Inc.

Use of this source code is governed by the [MIT License](https://opensource.org/licenses/MIT) that can be found in the LICENSE.txt file.
