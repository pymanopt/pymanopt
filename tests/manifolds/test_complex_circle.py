from pymanopt.manifolds import ComplexCircle

from .._test import TestCase


class TestComplexCircleManifold(TestCase):
    def setUp(self):
        self.dimension = 50
        self.manifold = ComplexCircle(self.dimension)

    def test_dim(self):
        self.assertEqual(self.manifold.dim, self.dimension)
