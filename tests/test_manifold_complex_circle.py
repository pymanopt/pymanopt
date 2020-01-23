import unittest

from pymanopt.manifolds import ComplexCircle


class TestComplexCircleManifold(unittest.TestCase):
    def setUp(self):
        self.dimension = 50
        self.man = ComplexCircle(self.dimension)

    def test_dim(self):
        self.assertEqual(self.man.dim, self.dimension)
