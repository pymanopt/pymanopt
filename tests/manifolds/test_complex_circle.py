from pymanopt.manifolds import ComplexCircle

from ._manifold_tests import ManifoldTestCase


class TestComplexCircleManifold(ManifoldTestCase):
    def setUp(self):
        self.dimension = 50
        self.manifold = ComplexCircle(self.dimension)

        super().setUp()

    def test_dim(self):
        self.assertEqual(self.manifold.dim, self.dimension)
