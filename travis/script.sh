#!/bin/bash

source "${0%/*}/setup.sh"

# Skip mypy test when priority is to make sure that other tests pass
# Mypy only works with Python 3
# if [ "${PYTHON_VERSION}" != "python2" ]; then
#   time mypy .
#   # Also test in python2 mode (but this is still in the python 3 CI
#   # instance, because mypy itself needs python 3)
#   time mypy --py2 .
# fi

if [[ $TRAVIS_OS_NAME == 'osx' ]]; then
  time python setup.py test
else
  # Test cases that need to run CoreML models won't work on Linux,
  # only run test cases that don't need it.
  time python setup.py test --addopts tests/graph_test.py
  time python setup.py test --addopts tests/custom_layers_test.py
fi
