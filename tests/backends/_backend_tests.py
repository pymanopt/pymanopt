import pytest

from pymanopt.manifolds.manifold import Manifold
import pymanopt.numerics as nx
from pymanopt.numerics import numpy_to_backend


def manifold_factory(*, point_layout):
    class CustomManifold(Manifold):
        def __init__(self):
            super().__init__(
                name="Test manifold", dimension=3, point_layout=point_layout
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


class TestUnaryFunction:
    """Test cost function, gradient and Hessian for a unary cost function.

    This test uses a cost function of the form::

        f = lambda x: nx.sum(x ** 2)
    """

    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        self.manifold = manifold_factory(point_layout=1)
        self.n = 10
        self.cost = None
        self.backend = 'numpy'

    def test_unary_function(self):
        cost = self.cost
        assert cost is not None
        n = self.n
        backend = self.backend

        x = numpy_to_backend(nx.random.normal(size=n), backend)

        # Test whether cost function accepts single argument.
        assert nx.allclose(nx.sum(x**2), cost(x))

        # Test whether gradient accepts single argument.
        euclidean_gradient = cost.get_gradient_operator()
        nx.allclose(2 * x, euclidean_gradient(x))

        # Test the Hessian.
        u = numpy_to_backend(nx.random.normal(size=n), backend)

        # Test whether Hessian accepts two regular arguments.
        ehess = cost.get_hessian_operator()
        # Test whether Hessian-vector product is correct.
        nx.allclose(2 * u, ehess(x, u))


class TestUnaryComplexFunction:
    """Test cost function, gradient and Hessian of complex unary cost function.

    This test uses a cost function of the form::

        f = lambda x: nx.sum(x ** 2).real
    """

    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        self.manifold = manifold_factory(point_layout=1)
        self.n = 10
        self.cost = None
        self.backend = 'numpy'

    def test_unary_function(self):
        cost = self.cost
        assert cost is not None
        n = self.n
        backend = self.backend

        x = numpy_to_backend(
            nx.random.normal(size=n) + 1j * nx.random.normal(size=n), backend)

        # Test whether cost function accepts single argument.
        assert nx.allclose(nx.sum(x**2).real, cost(x))

        # Test whether gradient accepts single argument.
        euclidean_gradient = cost.get_gradient_operator()
        nx.allclose(2 * x.conj(), euclidean_gradient(x))

        # Test the Hessian.
        u = numpy_to_backend(
            nx.random.normal(size=n) + 1j * nx.random.normal(size=n), backend)

        # Test whether Hessian accepts two regular arguments.
        ehess = cost.get_hessian_operator()
        # Test whether Hessian-vector product is correct.
        nx.allclose(2 * u.conj(), ehess(x, u))


class TestNaryFunction:
    """Test cost function, gradient and Hessian for an nary cost function.

    This test uses a cost function of the form::

        f = lambda x, y: x @ y

    This situation arises e.g. when optimizing over the
    :class:`pymanopt.manifolds.FixedRankEmbedded` manifold where points on the
    manifold are represented as a 3-tuple of a truncated SVD.
    """

    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        self.manifold = manifold_factory(point_layout=2)
        self.n = 10
        self.cost = None
        self.backend = 'numpy'

    def test_nary_function(self):
        cost = self.cost
        assert cost is not None
        n = self.n
        backend = self.backend

        x = numpy_to_backend(nx.random.normal(size=n), backend)
        y = numpy_to_backend(nx.random.normal(size=n), backend)

        assert nx.allclose(x @ y, cost(x, y))

        euclidean_gradient = cost.get_gradient_operator()
        g = euclidean_gradient(x, y)
        assert isinstance(g, (list, tuple))
        assert len(g) == 2
        for gi in g:
            assert isinstance(gi, nx.ndarray)
        g_x, g_y = g
        nx.allclose(g_x, y)
        nx.allclose(g_y, x)

        # Test the Hessian-vector product.
        u = numpy_to_backend(nx.random.normal(size=n), backend)
        v = numpy_to_backend(nx.random.normal(size=n), backend)

        ehess = cost.get_hessian_operator()
        h = ehess(x, y, u, v)
        assert isinstance(h, (list, tuple))
        assert len(h) == 2
        for hi in h:
            assert isinstance(hi, nx.ndarray)

        # Test whether the Hessian-vector product is correct.
        h_x, h_y = h
        nx.allclose(h_x, v)
        nx.allclose(h_y, u)


class TestNaryParameterGrouping:
    """Test parameter grouping for cost function, gradient and Hessian.

    This test assumes a cost function of the form::

        f = lambda x, y, z: nx.sum(x ** 2 * y + 3 * z)

    This situation could arise e.g. on product manifolds where one of the
    underlying manifolds represents points as a tuple of arrays.
    """

    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        self.manifold = manifold_factory(point_layout=3)
        self.n = 10
        self.cost = None
        self.backend = 'numpy'

    def test_nary_parameter_grouping(self):
        cost = self.cost
        assert cost is not None
        n = self.n
        backend = self.backend

        x, y, z = [
            numpy_to_backend(nx.random.normal(size=n), backend) for _ in range(3)]

        assert nx.allclose(nx.sum(x**2 + y + z**3), cost(x, y, z))

        euclidean_gradient = cost.get_gradient_operator()
        g = euclidean_gradient(x, y, z)

        assert isinstance(g, (list, tuple))
        assert len(g) == 3
        for grad in g:
            assert isinstance(grad, nx.ndarray)
        g_x, g_y, g_z = g

        # Verify correctness of the gradient.
        nx.allclose(g_x, 2 * x)
        nx.allclose(g_y, 1)
        nx.allclose(g_z, 3 * z**2)

        # Test the Hessian.
        u, v, w = [
            numpy_to_backend(nx.random.normal(size=n), backend) for _ in range(3)]

        ehess = cost.get_hessian_operator()
        h = ehess(x, y, z, u, v, w)

        # Test the type composition of the return value.
        assert isinstance(h, (list, tuple))
        assert len(h) == 3
        for hess in h:
            assert isinstance(hess, nx.ndarray)
        h_x, h_y, h_z = h

        # Test whether the Hessian-vector product is correct.
        nx.allclose(h_x, 2 * u)
        nx.allclose(h_y, 0)
        nx.allclose(h_z, 6 * z * w)


class TestVector:
    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        nx.seterr(all="raise")
        self.manifold = manifold_factory(point_layout=1)
        self.n = 15

    def test_compile(self):
        n = self.n
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=n), backend)

        correct_cost = nx.exp(nx.sum(Y**2))

        nx.allclose(correct_cost, self.cost(Y))

    def test_grad(self):
        n = self.n
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=n), backend)

        correct_grad = 2 * Y * nx.exp(nx.sum(Y**2))

        grad = self.cost.get_gradient_operator()

        nx.allclose(correct_grad, grad(Y))

    def test_hessian(self):
        n = self.n
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=n), backend)
        A = numpy_to_backend(nx.random.normal(size=n), backend)

        # First form hessian matrix H
        # Convert Y and A into matrices (row vectors)
        Ymat = Y[nx.newaxis, :]
        Amat = A[nx.newaxis, :]

        diag = nx.eye(n)

        H = nx.exp(nx.sum(Y**2)) * (4 * Ymat.T @ Ymat + 2 * diag)

        # Then 'left multiply' H by A
        correct_hess = nx.squeeze(nx.array(Amat @ H))

        hess = self.cost.get_hessian_operator()

        nx.allclose(correct_hess, hess(Y, A))


class TestMatrix:
    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        nx.seterr(all="raise")

        self.manifold = manifold_factory(point_layout=1)

        self.m = 10
        self.n = 15

    def test_compile(self):
        m, n = self.m, self.n
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=(m, n)), backend)

        correct_cost = nx.exp(nx.sum(Y**2))

        nx.allclose(correct_cost, self.cost(Y))

    def test_grad(self):
        m, n = self.m, self.n
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=(m, n)), backend)

        correct_grad = 2 * Y * nx.exp(nx.sum(Y**2))

        grad = self.cost.get_gradient_operator()

        nx.allclose(correct_grad, grad(Y))

    def test_hessian(self):
        m, n = self.m, self.n
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=(m, n)), backend)
        Y1 = Y.reshape(m, n, 1, 1)
        Y2 = Y.reshape(1, 1, m, n)
        A = numpy_to_backend(nx.random.normal(size=(m, n)), backend)

        # Create an m x n x m x n array with diag[i,j,k,l] == 1 iff
        # (i == k and j == l), this is a 'diagonal' tensor.
        diag = nx.eye(m * n).reshape(m, n, m, n)

        H = nx.exp(nx.sum(Y**2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = A.reshape(1, 1, m, n)

        correct_hess = nx.sum(H * Atensor, axis=(2, 3))

        hess = self.cost.get_hessian_operator()

        # Now test hess
        nx.allclose(correct_hess, hess(Y, A))


class TestTensor3:
    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        nx.seterr(all="raise")

        self.manifold = manifold_factory(point_layout=1)

        self.n1 = 3
        self.n2 = 4
        self.n3 = 5

    def test_compile(self):
        n1, n2, n3 = self.n1, self.n2, self.n3
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=(n1, n2, n3)), backend)

        correct_cost = nx.exp(nx.sum(Y**2))

        nx.allclose(correct_cost, self.cost(Y))

    def test_grad(self):
        n1, n2, n3 = self.n1, self.n2, self.n3
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=(n1, n2, n3)), backend)

        correct_grad = 2 * Y * nx.exp(nx.sum(Y**2))

        grad = self.cost.get_gradient_operator()

        nx.allclose(correct_grad, grad(Y))

    def test_hessian(self):
        n1, n2, n3 = self.n1, self.n2, self.n3
        backend = self.backend
        Y = numpy_to_backend(nx.random.normal(size=(n1, n2, n3)), backend)
        A = numpy_to_backend(nx.random.normal(size=(n1, n2, n3)), backend)

        # First form hessian tensor H (6th order)
        Y1 = Y.reshape(n1, n2, n3, 1, 1, 1)
        Y2 = Y.reshape(1, 1, 1, n1, n2, n3)

        # Create an n1 x n2 x n3 x n1 x n2 x n3 diagonal tensor
        diag = nx.eye(n1 * n2 * n3).reshape(n1, n2, n3, n1, n2, n3)

        H = nx.exp(nx.sum(Y**2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = A.reshape(1, 1, 1, n1, n2, n3)

        correct_hess = nx.sum(H * Atensor, axis=(3, 4, 5))

        hess = self.cost.get_hessian_operator()

        # Now test hess
        nx.allclose(correct_hess, hess(Y, A))


class TestMixed:
    @pytest.fixture(autouse=True)
    def initialize_test_case(self):
        nx.seterr(all="raise")

        self.manifold = manifold_factory(point_layout=3)

        n1 = self.n1 = 3
        n2 = self.n2 = 4
        n3 = self.n3 = 5
        n4 = self.n4 = 6
        n5 = self.n5 = 7
        n6 = self.n6 = 8

        self.y = y = (
            nx.random.normal(size=n1),
            nx.random.normal(size=(n2, n3)),
            nx.random.normal(size=(n4, n5, n6)),
        )
        self.a = a = (
            nx.random.normal(size=n1),
            nx.random.normal(size=(n2, n3)),
            nx.random.normal(size=(n4, n5, n6)),
        )

        self.correct_cost = (
            nx.exp(nx.sum(y[0] ** 2))
            + nx.exp(nx.sum(y[1] ** 2))
            + nx.exp(nx.sum(y[2] ** 2))
        )

        # Calculate correct grad
        g1 = 2 * y[0] * nx.exp(nx.sum(y[0] ** 2))
        g2 = 2 * y[1] * nx.exp(nx.sum(y[1] ** 2))
        g3 = 2 * y[2] * nx.exp(nx.sum(y[2] ** 2))

        self.correct_grad = (g1, g2, g3)

        # Calculate correct hess
        # 1. Vector
        Ymat = y[0][nx.newaxis, :]
        Amat = a[0][nx.newaxis, :]

        diag = nx.eye(n1)

        H = nx.exp(nx.sum(y[0] ** 2)) * (4 * Ymat.T @ Ymat + 2 * diag)

        # Then 'left multiply' H by A
        h1 = nx.array(Amat @ H).flatten()

        # 2. MATRIX
        # First form hessian tensor H (4th order)
        Y1 = y[1].reshape(n2, n3, 1, 1)
        Y2 = y[1].reshape(1, 1, n2, n3)

        # Create an m x n x m x n array with diag[i,j,k,l] == 1 iff
        # (i == k and j == l), this is a 'diagonal' tensor.
        diag = nx.eye(n2 * n3).reshape(n2, n3, n2, n3)

        H = nx.exp(nx.sum(y[1] ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = a[1].reshape(1, 1, n2, n3)

        h2 = nx.sum(H * Atensor, axis=(2, 3))

        # 3. Tensor3
        # First form hessian tensor H (6th order)
        Y1 = y[2].reshape(n4, n5, n6, 1, 1, 1)
        Y2 = y[2].reshape(1, 1, 1, n4, n5, n6)

        # Create an n1 x n2 x n3 x n1 x n2 x n3 diagonal tensor
        diag = nx.eye(n4 * n5 * n6).reshape(n4, n5, n6, n4, n5, n6)

        H = nx.exp(nx.sum(y[2] ** 2)) * (4 * Y1 * Y2 + 2 * diag)

        # Then 'right multiply' H by A
        Atensor = a[2].reshape(1, 1, 1, n4, n5, n6)

        h3 = nx.sum(H * Atensor, axis=(3, 4, 5))

        self.correct_hess = (h1, h2, h3)

    def test_compile(self):
        nx.allclose(self.correct_cost, self.cost(*self.y))

    def test_grad(self):
        grad = self.cost.get_gradient_operator()
        g = grad(*self.y)
        for k in range(len(g)):
            nx.allclose(self.correct_grad[k], g[k])

    def test_hessian(self):
        hess = self.cost.get_hessian_operator()

        # Now test hess
        h = hess(*self.y, *self.a)
        for k in range(len(h)):
            nx.allclose(self.correct_hess[k], h[k])
