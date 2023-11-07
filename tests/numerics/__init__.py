import functools
import numpy as np
from typing import Sequence

from pymanopt.numerics import NUMERICS_SUPPORTED_BACKENDS, numpy_to_backend


def _test_numerics_supported_backends(func):
    @functools.wraps(func)
    def wrapper(**kwargs):
        for backend in NUMERICS_SUPPORTED_BACKENDS:
            for key, value in kwargs.items():
                if type(value) == np.ndarray:
                    kwargs[key] = numpy_to_backend(value, backend)
                if isinstance(value, Sequence):
                    new_value = list()
                    for item in value:
                        if type(item) == np.ndarray:
                            new_value.append(numpy_to_backend(item, backend))
                        else:
                            new_value.append(item)
                    kwargs[key] = new_value

            func(**kwargs)

    return wrapper
