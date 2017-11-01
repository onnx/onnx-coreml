from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import click
from onnx import onnx_pb2
from onnx_coreml import convert


@click.command(
    help='convert ONNX model to CoreML model',
    context_settings={
        'help_option_names': ['-h', '--help']
    }
)
@click.argument('onnx_model', type=click.File('rb'))
@click.option('-o', '--output', required=True,
              type=str,
              help='Output path for the CoreML *.mlmodel file')
def onnx_to_coreml(onnx_model, output):
    onnx_model_proto = onnx_pb2.ModelProto()
    onnx_model_proto.ParseFromString(onnx_model.read())
    coreml_model = convert(onnx_model_proto)
    coreml_model.save(output)
