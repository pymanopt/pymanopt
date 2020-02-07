#!/bin/sh

set -o errexit

# With `pyenv` and its `pyenv-virtualenv` plugin installed, this script
# bootstraps a Python 3.7 development environment.

die() {
  echo $*
  exit 1
}

PYTHONVERSION="3.7.6"
NAME="pymanopt"
ENVNAME="$NAME-$PYTHONVERSION"

for p in pyenv pyenv-virtualenv; do
  command -v $p >/dev/null 2>&1 || { die "\`$p\` not found in PATH"; }
done

CONFIGURE_OPTS=--enable-shared pyenv install -s $PYTHONVERSION
pyenv virtualenv -f $PYTHONVERSION $ENVNAME
pyenv local $ENVNAME
pip install --upgrade pip setuptools wheel
pip install -r dev-requirements.txt
pip install -r requirements.txt
