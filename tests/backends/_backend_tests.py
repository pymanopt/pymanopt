from typing import Sequence, Union

import pytest
from numpy import testing as bk

from pymanopt.backends import Backend
from pymanopt.function import Function
from pymanopt.manifolds.manifold import Manifold


__all__ = [
    "manifold_factory",
    "TestUnaryFunction",
    "TestUnaryComplexFunction",
    "TestNaryFunction",
    "TestNaryParameterGrouping",
    "TestVector",
    "TestMatrix",
    "TestTensor3",
    "TestMixed",
]


def manifold_factory(
    *, point_layout: Union[int, Sequence[int]], backend: Backend
):
    class CustomManifold(Manifold):
        def __init__(self):
            super().__init__(
                name="Test manifold",
                dimension=3,
                point_layout=point_layout,
                backend=backend,
            )

        def _generic(self, *args, **kwargs):
            pass

        inner_product = _generic
        norm = _generic
        projection = _generic
        random_point = _generic
        random_tangent_vector = _generic
        zero_vector = _generic

    return CustomManifold()


class _Test:
    cost: Function
    manifold: Manifold


class TestUnaryFunction(_Test):
    """Test cost function, gradient and Hessian for a unary cost function.

    This test uses a cost function of the form::

        f = lambda x: bk.sum(x ** 2)
    """

    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 1
        self.n = 10

    def test_unary_function(self):
        cost = self.cost
        bk = self.manifold.backend
        n = self.n

        x = bk.random_normal(size=n)

        # Test whether cost function accepts single argument.
        bk.assert_allclose(bk.sum(x**2), self.cost(x))

        # Test whether gradient accepts single argument.
        euclidean_gradient = cost.get_gradient_operator()
        bk.assert_allclose(2 * x, euclidean_gradient(x))

        # Test the Hessian.
        u = bk.random_normal(size=n)

        # Test whether Hessian accepts two regular arguments.
        ehess = cost.get_hessian_operator()
        # Test whether Hessian-vector product is correct.
        bk.assert_allclose(2 * u, ehess(x, u))


class TestUnaryComplexFunction(_Test):
    """Test cost function, gradient and Hessian of complex unary cost function.

    This test uses a cost function of the form::

        f = lambda x: bk.sum(x ** 2).real
    """

    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 1
        self.n = 10

    def test_unary_function(self):
        cost = self.cost
        bk = self.manifold.backend
        n = self.n

        x = bk.random_normal(size=n)

        # Test whether cost function accepts single argument.
        bk.assert_allclose(bk.real(bk.sum(x**2)), cost(x))

        # Test whether gradient accepts single argument.
        euclidean_gradient = cost.get_gradient_operator()
        bk.assert_allclose(2 * bk.conjugate(x), euclidean_gradient(x))

        # Test the Hessian.
        u = bk.random_normal(size=n)

        # Test whether Hessian accepts two regular arguments.
        ehess = cost.get_hessian_operator()
        # Test whether Hessian-vector product is correct.
        bk.assert_allclose(2 * bk.conjugate(u), ehess(x, u))


class TestNaryFunction(_Test):
    """Test cost function, gradient and Hessian for an nary cost function.

    This test uses a cost function of the form::

        f = lambda x, y: x @ y

    This situation arises e.g. when optimizing over the
    :class:`pymanopt.manifolds.FixedRankEmbedded` manifold where points on the
    manifold are represented as a 3-tuple of a truncated SVD.
    """

    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 2
        self.n = 10

    def test_nary_function(self):
        cost = self.cost
        bk = self.manifold.backend
        n = self.n

        x = bk.random_normal(size=n)
        y = bk.random_normal(size=n)

        bk.assert_allclose(x @ y, cost(x, y))

        euclidean_gradient = cost.get_gradient_operator()
        g = euclidean_gradient(x, y)
        assert isinstance(g, (list, tuple))
        assert len(g) == 2
        for gi in g:
            assert isinstance(gi, bk.array_t)
        g_x, g_y = g
        bk.assert_allclose(g_x, y)
        bk.assert_allclose(g_y, x)

        # Test the Hessian-vector product.
        u = bk.random_normal(size=n)
        v = bk.random_normal(size=n)

        ehess = cost.get_hessian_operator()
        h = ehess(x, y, u, v)
        assert isinstance(h, (list, tuple))
        assert len(h) == 2
        for hi in h:
            assert isinstance(hi, bk.array_t)

        # Test whether the Hessian-vector product is correct.
        h_x, h_y = h
        bk.assert_allclose(h_x, v)
        bk.assert_allclose(h_y, u)


class TestNaryParameterGrouping(_Test):
    """Test parameter grouping for cost function, gradient and Hessian.

    This test assumes a cost function of the form::

        f = lambda x, y, z: bk.sum(x ** 2 * y + 3 * z)

    This situation could arise e.g. on product manifolds where one of the
    underlying manifolds represents points as a tuple of arrays.
    """

    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 3
        self.n = 10

    def test_nary_parameter_grouping(self):
        cost = self.cost
        bk = self.manifold.backend
        n = self.n

        x, y, z = [bk.random_normal(size=n) for _ in range(3)]

        bk.assert_allclose(bk.sum(x**2 + y + z**3), cost(x, y, z))

        euclidean_gradient = cost.get_gradient_operator()
        g = euclidean_gradient(x, y, z)

        assert isinstance(g, (list, tuple))
        assert len(g) == 3
        for grad in g:
            assert isinstance(grad, bk.array_t), f"actual type: {type(grad)}"
        g_x, g_y, g_z = g

        # Verify correctness of the gradient.
        bk.assert_allclose(g_x, 2 * x)
        bk.assert_allclose(g_y, 1)
        bk.assert_allclose(g_z, 3 * z**2)

        # Test the Hessian.
        u, v, w = [bk.random_normal(size=n) for _ in range(3)]

        ehess = cost.get_hessian_operator()
        h = ehess(x, y, z, u, v, w)

        # Test the type composition of the return value.
        assert isinstance(h, (list, tuple))
        assert len(h) == 3
        for hess in h:
            assert isinstance(hess, bk.array_t), f"actual type: {type(hess)}"
        h_x, h_y, h_z = h

        # Test whether the Hessian-vector product is correct.
        bk.assert_allclose(h_x, 2 * u)
        bk.assert_allclose(h_y, 0.0)
        bk.assert_allclose(h_z, 6 * z * w)


class TestVector(_Test):
    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 1
        self.n = 15

    @pytest.fixture(autouse=True)
    def post_setup(self, setup):
        bk = self.manifold.backend
        n = self.n
        # bk.seterr(all="raise")

        Y = self.Y = bk.random_normal(size=n)
        A = self.A = bk.random_normal(size=n)

        # Calculate correct cost and grad...
        self.correct_cost = bk.exp(bk.sum(Y**2))
        self.correct_grad = 2 * Y * bk.exp(bk.sum(Y**2))

        # ... and hess
        # First form hessian matrix H
        # Convert Y and A into matrices (row vectors)
        Ymat = Y[bk.newaxis, :]
        Amat = A[bk.newaxis, :]

        diag = bk.eye(n)

        H = bk.exp(bk.sum(Y**2)) * (4 * Ymat.T @ Ymat + 2 * diag)

        # Then 'left multiply' H by A
        self.correct_hess = bk.squeeze(bk.array(Amat @ H))

    def test_compile(self):
        bk.assert_allclose(self.correct_cost, self.cost(self.Y))

    def test_grad(self):
        grad = self.cost.get_gradient_operator()
        bk.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.cost.get_hessian_operator()

        # Now test hess
        bk.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestMatrix(_Test):
    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 1
        self.m = 10
        self.n = 15

    @pytest.fixture(autouse=True)
    def post_setup(self, setup):
        bk = self.manifold.backend
        m = self.m
        n = self.n
        # bk.seterr(all="raise")

        Y = self.Y = bk.random_normal(size=(m, n))
        A = self.A = bk.random_normal(size=(m, n))

        # Calculate correct cost and grad...
        self.correct_cost = bk.exp(bk.sum(Y**2))
        self.correct_grad = 2 * Y * bk.exp(bk.sum(Y**2))

        # ... and hess
        # First form hessian tensor H (4th order)
        Y1 = Y.reshape(m, n, 1, 1)
        Y2 = Y.reshape(1, 1, m, n)

        # Create an m x n x m x n array with diag[i,j,k,l] == 1 iff
        # (i == k and j == l), this is a 'diagonal' tensor.
        diag = bk.eye(m * n).reshape(m, n, m, n)

        H = bk.exp(bk.sum(Y**2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = A.reshape(1, 1, m, n)

        self.correct_hess = bk.sum(H * Atensor, axis=(2, 3))

    def test_compile(self):
        bk.assert_allclose(self.correct_cost, self.cost(self.Y))

    def test_grad(self):
        grad = self.cost.get_gradient_operator()
        bk.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.cost.get_hessian_operator()

        # Now test hess
        bk.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestTensor3(_Test):
    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 1
        self.n1 = 3
        self.n2 = 4
        self.n3 = 5

    @pytest.fixture(autouse=True)
    def post_setup(self, setup):
        bk = self.manifold.backend
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        # bk.seterr(all="raise")

        Y = self.Y = bk.random_normal(size=(n1, n2, n3))
        A = self.A = bk.random_normal(size=(n1, n2, n3))

        # Calculate correct cost and grad...
        self.correct_cost = bk.exp(bk.sum(Y**2))
        self.correct_grad = 2 * Y * bk.exp(bk.sum(Y**2))

        # ... and hess
        # First form hessian tensor H (6th order)
        Y1 = Y.reshape(n1, n2, n3, 1, 1, 1)
        Y2 = Y.reshape(1, 1, 1, n1, n2, n3)

        # Create an n1 x n2 x n3 x n1 x n2 x n3 diagonal tensor
        diag = bk.eye(n1 * n2 * n3).reshape(n1, n2, n3, n1, n2, n3)

        H = bk.exp(bk.sum(Y**2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = A.reshape(1, 1, 1, n1, n2, n3)

        self.correct_hess = bk.sum(H * Atensor, axis=(3, 4, 5))

    def test_compile(self):
        bk.assert_allclose(self.correct_cost, self.cost(self.Y))

    def test_grad(self):
        grad = self.cost.get_gradient_operator()
        bk.assert_allclose(self.correct_grad, grad(self.Y))

    def test_hessian(self):
        hess = self.cost.get_hessian_operator()

        # Now test hess
        bk.assert_allclose(self.correct_hess, hess(self.Y, self.A))


class TestMixed(_Test):
    @pytest.fixture(autouse=True)
    def pre_setup(self):
        self.point_layout = 3
        self.n1 = 3
        self.n2 = 4
        self.n3 = 5
        self.n4 = 6
        self.n5 = 7
        self.n6 = 8

    @pytest.fixture(autouse=True)
    def post_setup(self, setup):
        bk = self.manifold.backend
        n1 = self.n1
        n2 = self.n2
        n3 = self.n3
        n4 = self.n4
        n5 = self.n5
        n6 = self.n6
        # bk.seterr(all="raise")

        self.y = y = (
            bk.random_normal(size=n1),
            bk.random_normal(size=(n2, n3)),
            bk.random_normal(size=(n4, n5, n6)),
        )
        self.a = a = (
            bk.random_normal(size=n1),
            bk.random_normal(size=(n2, n3)),
            bk.random_normal(size=(n4, n5, n6)),
        )

        self.correct_cost = (
            bk.exp(bk.sum(y[0] ** 2))
            + bk.exp(bk.sum(y[1] ** 2))
            + bk.exp(bk.sum(y[2] ** 2))
        )

        # Calculate correct grad
        g1 = 2 * y[0] * bk.exp(bk.sum(y[0] ** 2))
        g2 = 2 * y[1] * bk.exp(bk.sum(y[1] ** 2))
        g3 = 2 * y[2] * bk.exp(bk.sum(y[2] ** 2))

        self.correct_grad = (g1, g2, g3)

        # Calculate correct hess
        # 1. Vector
        Ymat = y[0][bk.newaxis, :]
        Amat = a[0][bk.newaxis, :]

        diag = bk.eye(n1)

        H = bk.exp(bk.sum(y[0] ** 2)) * (4 * Ymat.T @ Ymat + 2 * diag)

        # Then 'left multiply' H by A
        h1 = bk.array(Amat @ H).flatten()

        # 2. MATRIX
        # First form hessian tensor H (4th order)
        Y1 = y[1].reshape(n2, n3, 1, 1)
        Y2 = y[1].reshape(1, 1, n2, n3)

        # Create an m x n x m x n array with diag[i,j,k,l] == 1 iff
        # (i == k and j == l), this is a 'diagonal' tensor.
        diag = bk.eye(n2 * n3).reshape(n2, n3, n2, n3)

        H = bk.exp(bk.sum(y[1] ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = a[1].reshape(1, 1, n2, n3)

        h2 = bk.sum(H * Atensor, axis=(2, 3))

        # 3. Tensor3
        # First form hessian tensor H (6th order)
        Y1 = y[2].reshape(n4, n5, n6, 1, 1, 1)
        Y2 = y[2].reshape(1, 1, 1, n4, n5, n6)

        # Create an n1 x n2 x n3 x n1 x n2 x n3 diagonal tensor
        diag = bk.eye(n4 * n5 * n6).reshape(n4, n5, n6, n4, n5, n6)

        H = bk.exp(bk.sum(y[2] ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = a[2].reshape(1, 1, 1, n4, n5, n6)

        h3 = bk.sum(H * Atensor, axis=(3, 4, 5))

        self.correct_hess = (h1, h2, h3)

    def test_compile(self):
        bk.assert_allclose(self.correct_cost, self.cost(*self.y))

    def test_grad(self):
        grad = self.cost.get_gradient_operator()
        g = grad(*self.y)
        for k in range(len(g)):
            bk.assert_allclose(self.correct_grad[k], g[k])

    def test_hessian(self):
        hess = self.cost.get_hessian_operator()

        # Now test hess
        h = hess(*self.y, *self.a)
        for k in range(len(h)):
            bk.assert_allclose(self.correct_hess[k], h[k])
