from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import numpy as np
from typing import Any, Sequence, List
from onnx.backend.base import BackendRep, namedtupledict
from coremltools.models import MLModel  #type: ignore
from typing import Dict, Any, Text


class CoreMLRep(BackendRep):
    def __init__(self,
                 coreml_model,  # type: MLModel
                 onnx_outputs,  # type: Dict[Text, Any]
                 useCPUOnly=False,  # type: bool
                 ):
        # type: (...) -> None
        super(CoreMLRep, self).__init__()
        self.model = coreml_model
        self.useCPUOnly = useCPUOnly

        spec = coreml_model.get_spec()
        self.input_names = [str(i.name) for i in spec.description.input]
        self.output_names = [str(o.name) for o in spec.description.output]
        self.onnx_outputs = onnx_outputs #{str:Tuple}, i.e. {name:shape}

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
                output_values[i] = np.reshape(output_, self.onnx_outputs[self.output_names[i]]) #type: ignore
            except RuntimeError:
                print("Output '%s' shape incompatible between CoreML (%s) and onnx (%s)"
                      %(self.output_names[i], output_.shape,
                        self.onnx_outputs[self.output_names[i]]))
        return namedtupledict('Outputs',
                              self.output_names)(*output_values)
