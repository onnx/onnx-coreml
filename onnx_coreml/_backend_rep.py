from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import numpy as np

from onnx.backend.base import BackendRep, namedtupledict


class CoreMLRep(BackendRep):
    def __init__(self, coreml_model, useCPUOnly=False):
        super(CoreMLRep, self).__init__()
        self.model = coreml_model
        self.useCPUOnly = useCPUOnly

        spec = coreml_model.get_spec()
        self.input_names = [str(i.name) for i in spec.description.input]
        self.output_names = [str(o.name) for o in spec.description.output]

    def run(self, inputs, **kwargs):
        super(CoreMLRep, self).run(inputs, **kwargs)
        inputs_ = inputs
        for i, input_ in enumerate(inputs_):
            shape = input_.shape
            if len(shape) == 4:
                # reshape to [seq, batch, channels, height, width]
                inputs_[i] = input_[np.newaxis, :]
        input_dict = dict(zip(self.input_names, inputs_))
        prediction = self.model.predict(input_dict, self.useCPUOnly)
        output_values = [prediction[name] for name in self.output_names]
        return namedtupledict('Outputs',
                              self.output_names)(*output_values)
