__all__ = ["__version__", "function", "manifolds", "optimizers", "Problem"]

import os

from pymanopt import function, manifolds, optimizers
from pymanopt.core.problem import Problem

from . import _version


__version__ = _version.get_versions()["version"]

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.getenv("TF_CPP_MIN_LOG_LEVEL", "2")
