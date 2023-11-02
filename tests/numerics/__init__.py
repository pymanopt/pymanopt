import functools
import numpy as np

from pymanopt.numerics import NUMERICS_SUPPORTED_BACKENDS, numpy_to_backend


def _test_numerics_supported_backends(func):
    @functools.wraps(func)
    def wrapper(**kwargs):
        for backend in NUMERICS_SUPPORTED_BACKENDS:
            for key, value in kwargs.items():
                if type(value) == np.ndarray:
                    kwargs[key] = numpy_to_backend(value, backend)
            func(**kwargs)

    return wrapper
