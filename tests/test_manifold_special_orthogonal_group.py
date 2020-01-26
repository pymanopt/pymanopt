import unittest

from pymanopt.manifolds import SpecialOrthogonalGroup


class TestSpecialOrthogonalGroup(unittest.TestCase):
    def test_constructor(self):
        SpecialOrthogonalGroup(10, 3)
