from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import numpy as np
from typing import Any, Sequence, List
from onnx.backend.base import BackendRep, namedtupledict
from coremltools.models import MLModel  #type: ignore
from typing import Dict, Any, Text, Tuple
from onnx import TensorProto
from ._graph import EdgeInfo

class CoreMLRep(BackendRep):
    def __init__(self,
                 coreml_model,  # type: MLModel
                 onnx_outputs_info,  # type: Dict[Text, EdgeInfo]
                 useCPUOnly=False,  # type: bool
                 ):
        # type: (...) -> None
        super(CoreMLRep, self).__init__()
        self.model = coreml_model
        self.useCPUOnly = useCPUOnly

        spec = coreml_model.get_spec()
        self.input_names = [str(i.name) for i in spec.description.input]
        self.output_names = [str(o.name) for o in spec.description.output]
        self.onnx_outputs_info = onnx_outputs_info  # type: Dict[Text, EdgeInfo]

    def run(self,
            inputs,  # type: Any
            **kwargs  # type: Any
            ):
        # type: (...) -> namedtupledict
        super(CoreMLRep, self).run(inputs, **kwargs)
        inputs_ = inputs
        _reshaped = False
        for i, input_ in enumerate(inputs_):
            shape = input_.shape
            if len(shape) == 4 or len(shape) == 2:
                inputs_[i] = input_[np.newaxis, :]
                _reshaped = True
        input_dict = dict(
            zip(self.input_names,
                map(np.array, inputs_)))
        prediction = self.model.predict(input_dict, self.useCPUOnly)
        output_values = [prediction[name] for name in self.output_names]
        for i, output_ in enumerate(output_values):
            shape = output_.shape
            #reshape the CoreML output to match Onnx's in shape
            try:
                output_values[i] = np.reshape(output_, self.onnx_outputs_info[self.output_names[i]][2])  # type: ignore
                if self.onnx_outputs_info[self.output_names[i]][1] == TensorProto.FLOAT:
                    output_values[i] = output_values[i].astype(np.float32)
            except RuntimeError:
                print("Output '%s' shape incompatible between CoreML (%s) and onnx (%s)"
                      %(self.output_names[i], output_.shape,
                        self.onnx_outputs[self.output_names[i]]))
        return namedtupledict('Outputs',
                              self.output_names)(*output_values)
