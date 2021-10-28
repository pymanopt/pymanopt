#!/bin/sh

set -o errexit

# With `pyenv` and its `pyenv-virtualenv` plugin installed, this script
# bootstraps a Python 3.7 development environment.

die() {
  echo $*
  exit 1
}

for p in pyenv; do
  command -v $p >/dev/null 2>&1 || { die "\`$p\` not found in PATH"; }
done

CONFIGURE_OPTS=--enable-shared pyenv install 3.7.10
pyenv virtualenv 3.7.10 pymanopt-3.7.10
pip install pip==21.3.1 setuptools==47.1.0 wheel==0.37.0
pip install -r requirements/base.txt -r requirements/dev.txt
