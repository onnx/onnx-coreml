#!/bin/bash

# Don't source setup.sh here, because the virtualenv might not be set up yet

# Install dependencies
brew install ccache protobuf

# Install Python.
# The brew update/upgrade commands are needed because otherwise homebrew
# exits with an error code when installing python2 since it is already installed
# but with a different version.
if [ "${PYTHON_VERSION}" != "python2" ]; then
  brew install ${PYTHON_VERSION}
fi

pip install virtualenv
virtualenv -p /usr/local/bin/${PYTHON_VERSION} "${HOME}/virtualenv"
source "${HOME}/virtualenv/bin/activate"
python --version

# Update all existing python packages
pip list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1  | xargs -n1 pip install -U
