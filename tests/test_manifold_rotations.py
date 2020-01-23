import unittest

from pymanopt.manifolds import Rotations


class TestRotationsManifold(unittest.TestCase):
    def test_constructor(self):
        Rotations(10, 3)
