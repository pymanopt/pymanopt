import pytest
from copy import deepcopy

from pymanopt.numerics import NUMERICS_SUPPORTED_BACKENDS


class BackendManifold(type):
    def __new__(cls, name, bases, attrs):
        # decorate all tests with the backend setup
        keys = list(attrs.keys())
        for key in keys:
            if key.startswith("test_"):
                test = deepcopy(attrs[key])
                for backend in NUMERICS_SUPPORTED_BACKENDS:
                    attrs[f"{key}_{backend}"] = cls.test_with_backends(cls, test, backend)
                del attrs[key]

        return super().__new__(cls, name, bases, attrs)

    def test_with_backends(cls, test, backend):
        def wrapped_test(self, *args, **kwargs):
            self.manifold._backend = backend
            test(self, *args, **kwargs)
        return wrapped_test
