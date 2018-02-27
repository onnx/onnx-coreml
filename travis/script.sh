#!/bin/bash

source "${0%/*}/setup.sh"

# Mypy only works with Python 3
if [ "${PYTHON_VERSION}" != "python2" ]; then
  time mypy .
  # TODO Also test in python2 mode (but this is still in the python 3 CI
  # instance, because mypy itself needs python 3)
  # time mypy --py2 .
fi

#time python setup.py test
time python tests/transformers_test.py
time python tests/convert_test.py
time python test/graph_test.py
time python test/transformers_test.py
time python test/operators_test.py
time python test/onnx_backend_models_test.py 

