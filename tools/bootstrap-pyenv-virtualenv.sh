#!/bin/sh

set -o errexit

# With `pyenv` and its `pyenv-virtualenv` plugin installed, this script
# bootstraps a Python development environment.

die() {
  echo "$@"
  exit 1
}

for p in pyenv pyenv-virtualenv; do
  command -v $p >/dev/null 2>&1 || { die "\`$p\` not found in PATH"; }
done

PYTHON_VERSION="3.9.9"
NAME="pymanopt"
ENV_NAME="$NAME-$PYTHON_VERSION"

CONFIGURE_OPTS=--enable-shared pyenv install -s "$PYTHON_VERSION"
pyenv virtualenv -f "$PYTHON_VERSION" "$ENV_NAME"
pyenv local "$ENV_NAME"
pip install -r requirements/ci.txt
pip install \
  -r requirements/base.txt \
  -r requirements/dev.txt \
  -r requirements/docs.txt
