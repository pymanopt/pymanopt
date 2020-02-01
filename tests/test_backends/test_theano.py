import numpy.testing as np_testing
import theano.tensor as T

from pymanopt.function import Theano
from . import _backend_tests


class TestUnaryFunction(_backend_tests.TestUnaryFunction):
    def setUp(self):
        super().setUp()

        x = T.vector()

        @Theano(x)
        def cost(x):
            return T.sum(x ** 2)

        self.cost = cost


class TestNaryFunction(_backend_tests.TestNaryFunction):
    def setUp(self):
        super().setUp()

        x = T.vector()
        y = T.vector()

        @Theano(x, y)
        def cost(x, y):
            return T.dot(x, y)

        self.cost = cost


class TestNaryParameterGrouping(_backend_tests.TestNaryParameterGrouping):
    def setUp(self):
        super().setUp()

        x = T.vector()
        y = T.vector()
        z = T.vector()

        @Theano(x, y, z)
        def cost(x, y, z):
            return T.sum(x ** 2 + y + z ** 3)

        self.cost = cost


class TestVector(_backend_tests.TestVector):
    def setUp(self):
        super().setUp()

        X = T.vector()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        self.cost = cost

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def new_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        X = T.vector()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        # And check that all is still well
        hess = cost.compute_hessian_vector_product()

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

        # Fix broken Rop
        T.exp.R_op = Rop


class TestMatrix(_backend_tests.TestMatrix):
    def setUp(self):
        super().setUp()

        X = T.matrix()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        self.cost = cost

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def broken_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = broken_Rop

        # Rebuild graph to force recompile
        X = T.matrix()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        # And check that all is still well
        hess = cost.compute_hessian_vector_product()

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

        # Fix broken Rop
        T.exp.R_op = Rop


class TestTensor3(_backend_tests.TestTensor3):
    def setUp(self):
        super().setUp()

        X = T.tensor3()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        self.cost = cost

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def new_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        X = T.tensor3()

        @Theano(X)
        def cost(X):
            return T.exp(T.sum(X ** 2))

        # And check that all is still well
        hess = cost.compute_hessian_vector_product()

        np_testing.assert_allclose(self.correct_hess, hess(self.Y, self.A))

        # Fix broken Rop
        T.exp.R_op = Rop


class TestMixed(_backend_tests.TestMixed):
    def setUp(self):
        super().setUp()

        x = T.vector()
        y = T.matrix()
        z = T.tensor3()

        @Theano(x, y, z)
        def cost(x, y, z):
            return (T.exp(T.sum(x ** 2)) +
                    T.exp(T.sum(y ** 2)) +
                    T.exp(T.sum(z ** 2)))

        self.cost = cost

    def test_hessian_no_Rop(self):
        # Break the Rop in T.exp
        Rop = T.exp.R_op

        def new_Rop(x, y):
            raise NotImplementedError
        T.exp.R_op = new_Rop

        # Rebuild graph to force recompile
        x = T.vector()
        y = T.matrix()
        z = T.tensor3()
        f = T.exp(T.sum(x ** 2)) + T.exp(T.sum(y ** 2)) + T.exp(T.sum(z ** 2))

        # Alternative use of `Theano' in decorator notation.
        cost = Theano(x, y, z)(lambda x, y, z: f)

        # And check that all is still well
        hess = cost.compute_hessian_vector_product()

        h = hess(*self.y, *self.a)
        for k in range(len(h)):
            np_testing.assert_allclose(self.correct_hess[k], h[k])

        # Fix broken Rop
        T.exp.R_op = Rop
