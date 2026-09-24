import pytest

from pymanopt.backends.numpy_backend import NumpyBackend
from pymanopt.manifolds import ComplexCircle


class TestComplexCircleManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dimension = 50
        self.manifold = ComplexCircle(self.dimension, backend=NumpyBackend())

    def test_dim(self):
        assert self.manifold.dim == self.dimension
