import functools
import numpy as np
from typing import Union, Sequence

from pymanopt.numerics import NUMERICS_SUPPORTED_BACKENDS, to_backend


def decorator_test_numerics_supported_backends(
    backends=NUMERICS_SUPPORTED_BACKENDS
):
    """Decorator to test a function with all supported backends."""
    if not isinstance(backends, Union[tuple, list]):
        backends = [backends]

    def decorator(func):
        @functools.wraps(func)
        def wrapper(**kwargs):
            for backend in backends:
                for key, value in kwargs.items():
                    if type(value) == np.ndarray:
                        kwargs[key] = to_backend(value, backend)
                    if isinstance(value, Sequence):
                        new_value = list()
                        for item in value:
                            if type(item) == np.ndarray:
                                new_value.append(to_backend(item, backend))
                            else:
                                new_value.append(item)
                        kwargs[key] = new_value

                func(**kwargs)

        return wrapper

    return decorator
