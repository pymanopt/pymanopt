import pytest

from pymanopt.manifolds import ComplexCircle


class TestComplexCircleManifold:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.dimension = 50
        self.manifold = ComplexCircle(self.dimension)

    def test_dim(self):
        assert self.manifold.dim == self.dimension
