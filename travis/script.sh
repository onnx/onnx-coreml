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
time python setup.py test -m tests.transformers_test
time python setup.py test -m tests.convert_test
time python setup.py test -m tests.graph_test
time python setup.py test -m tests.operators_test
time python setup.py test -m tests.onnx_backend_models_test

