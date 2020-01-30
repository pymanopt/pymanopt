from pymanopt.manifolds import SpecialOrthogonalGroup
from .._test import TestCase


class TestSpecialOrthogonalGroup(TestCase):
    def test_constructor(self):
        SpecialOrthogonalGroup(10, 3)
