set -ex

export top_dir=$(dirname ${0%/*})

source "${HOME}/virtualenv/bin/activate"
python --version

# setup ccache
export PATH="/usr/local/opt/ccache/libexec:$PATH"
ccache --max-size 1G
