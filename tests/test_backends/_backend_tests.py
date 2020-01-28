import unittest

import numpy as np
from numpy import random as rnd, testing as np_testing


class TestUnaryFunction(unittest.TestCase):
    """Test cost function, gradient and Hessian for a simple unary function.
    """

    def setUp(self):
        self.n = 10
        self.cost = None

    def test_unary_function(self):
        cost = self.cost
        assert cost is not None
        n = self.n

        x = rnd.randn(n)

        # Test whether cost function accepts single argument.
        self.assertAlmostEqual(np.sum(x ** 2), cost(x))

        # Test whether gradient accepts single argument.
        egrad = cost.compute_gradient()
        np_testing.assert_allclose(2 * x, egrad(x))

        # Test the Hessian.
        u = rnd.randn(self.n)

        # Test whether Hessian accepts two regular arguments.
        ehess = cost.compute_hessian()
        # Test whether Hessian-vector product is correct.
        np_testing.assert_allclose(2 * u, ehess(x, u))


class TestNaryFunction(unittest.TestCase):
    """Test cost function, gradient and Hessian for cost functions accepting
    multiple arguments. This situation arises e.g. when optimizing over the
    FixedRankEmbedded manifold where points on the manifold are represented as
    a 3-tuple making up a truncated SVD.
    """

    def setUp(self):
        self.n = 10
        self.cost = None

    def test_nary_function(self):
        cost = self.cost
        assert cost is not None
        n = self.n

        x = rnd.randn(n)
        y = rnd.randn(n)

        # The argument signature of the cost function implies we are NOT on the
        # product manifold so solvers would call the wrapped cost function with
        # one argument, in this case a tuple of vectors.
        self.assertAlmostEqual(np.dot(x, y), cost((x, y)))

        egrad = cost.compute_gradient()
        g = egrad((x, y))
        # Since we treat the tuple (x, y) as one argument, we expect the result
        # of a call to the gradient function to be a tuple with two elements.
        self.assertIsInstance(g, (list, tuple))
        self.assertEqual(len(g), 2)
        for gi in g:
            self.assertIsInstance(gi, np.ndarray)
        g_x, g_y = g
        np_testing.assert_allclose(g_x, y)
        np_testing.assert_allclose(g_y, x)

        # Test the Hessian-vector product.
        u = rnd.randn(n)
        v = rnd.randn(n)

        ehess = cost.compute_hessian()
        h = ehess((x, y), (u, v))
        self.assertIsInstance(h, (list, tuple))
        self.assertEqual(len(h), 2)
        for hi in h:
            self.assertIsInstance(hi, np.ndarray)

        # Test whether the Hessian-vector product is correct.
        h_x, h_y = h
        np_testing.assert_allclose(h_x, v)
        np_testing.assert_allclose(h_y, u)


class TestNaryParameterGrouping(unittest.TestCase):
    """Test cost function, gradient and Hessian for a complex cost function one
    would define on product manifolds where one of the underlying manifolds
    represents points as a tuple of numpy.ndarrays.
    """

    def setUp(self):
        self.n = 10
        self.cost = None

    def test_nary_parameter_grouping(self):
        cost = self.cost
        assert cost is not None
        n = self.n

        x = rnd.randn(n)
        y = rnd.randn(n)
        z = rnd.randn(n)

        # The signature of the cost function now implies that we are on the
        # product manifold, so we mimic the behavior of solvers by calling the
        # cost function with a single argument: a tuple containing a tuple (x,
        # y) and a single vector z.
        self.assertAlmostEqual(np.sum(x ** 2 + y + z ** 3), cost(((x, y), z)))

        egrad = cost.compute_gradient()
        g = egrad(((x, y), z))

        # We defined the cost function signature to treat the first two
        # arguments as one parameter, so a call to the gradient must produce
        # two elements.
        self.assertIsInstance(g, (list, tuple))
        self.assertEqual(len(g), 2)
        g_xy, g_z = g
        self.assertIsInstance(g_xy, (list, tuple))
        self.assertEqual(len(g_xy), 2)
        self.assertIsInstance(g_z, np.ndarray)

        # Verify correctness of the gradient.
        np_testing.assert_allclose(g_xy[0], 2 * x)
        np_testing.assert_allclose(g_xy[1], 1)
        np_testing.assert_allclose(g_z, 3 * z ** 2)

        # Test the Hessian.
        u = rnd.randn(n)
        v = rnd.randn(n)
        w = rnd.randn(n)

        ehess = cost.compute_hessian()
        h = ehess(((x, y), z), ((u, v), w))

        # Test the type composition of the return value.
        self.assertIsInstance(h, (list, tuple))
        self.assertEqual(len(h), 2)
        h_xy, h_z = h
        self.assertIsInstance(h_xy, (list, tuple))
        self.assertEqual(len(h_xy), 2)
        self.assertIsInstance(h_z, np.ndarray)

        # Test whether the Hessian-vector product is correct.
        np_testing.assert_allclose(h_xy[0], 2 * u)
        np_testing.assert_allclose(h_xy[1], 0)
        np_testing.assert_allclose(h_z, 6 * z * w)
