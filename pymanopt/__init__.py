__all__ = ["__version__", "function", "Problem"]

import os

from pymanopt import function
from pymanopt._version import __version__
from pymanopt.core.problem import Problem


os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get(
    "TF_CPP_MIN_LOG_LEVEL", "2"
)
