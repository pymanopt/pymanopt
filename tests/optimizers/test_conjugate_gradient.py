from pymanopt.optimizers import ConjugateGradient

from .._test import TestCase


class TestConjugateGradient(TestCase):
    def test_beta_type(self):
        with self.assertRaises(ValueError):
            ConjugateGradient(beta_rule="SomeUnknownBetaRule")
