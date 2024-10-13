import pytest

from pymanopt.manifolds import SpecialOrthogonalGroup, UnitaryGroup
from pymanopt.numerics import NumpyNumericsBackend


class TestSpecialOrthogonalGroup:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = n = 10
        self.k = k = 3
        self.backend = NumpyNumericsBackend()
        self.so_product = SpecialOrthogonalGroup(
            n, k=k, backend=NumpyNumericsBackend()
        )
        self.so = SpecialOrthogonalGroup(n, backend=NumpyNumericsBackend())
        self.so_polar = SpecialOrthogonalGroup(
            n, retraction="polar", backend=NumpyNumericsBackend()
        )
        self.manifold = self.so

    def test_random_point(self):
        point = self.so.random_point()
        assert point.shape == (self.n, self.n)
        self.backend.assert_allclose(point.T @ point - point @ point.T, 0)
        self.backend.assert_allclose(point.T @ point, self.backend.eye(self.n))
        self.backend.assert_allclose(self.backend.linalg_det(point), 1)

    def test_random_point_product(self):
        point = self.so_product.random_point()
        assert point.shape == (self.k, self.n, self.n)
        self.backend.assert_allclose(
            self.backend.transpose(point) @ point
            - point @ self.backend.transpose(point),
            0,
        )
        self.backend.assert_allclose(
            self.backend.transpose(point) @ point,
            self.backend.multieye(self.k, self.n),
        )
        self.backend.assert_allclose(self.backend.linalg_det(point), 1)

    def test_random_tangent_vector(self):
        point = self.so.random_point()
        tangent_vector = self.so.random_tangent_vector(point)
        self.backend.assert_allclose(tangent_vector, -tangent_vector.T)

    def test_random_tangent_vector_product(self):
        point = self.so_product.random_point()
        tangent_vector = self.so_product.random_tangent_vector(point)
        self.backend.assert_allclose(
            tangent_vector, -self.backend.transpose(tangent_vector)
        )

    @pytest.mark.parametrize("manifold_attribute", ["so", "so_polar"])
    def test_retraction(self, manifold_attribute):
        manifold = getattr(self, manifold_attribute)

        # Test that the result is on the manifold.
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        xretru = manifold.retraction(x, u)

        self.backend.assert_allclose(xretru.T @ xretru - xretru @ xretru.T, 0)
        self.backend.assert_allclose(
            xretru.T @ xretru, self.backend.eye(xretru.shape[-1])
        )
        self.backend.assert_allclose(self.backend.linalg_det(xretru), 1)

    def test_exp_log_inverse(self):
        s = self.so
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        self.backend.assert_allclose(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.so
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        self.backend.assert_allclose(U, Ulogexp)


class TestUnitaryGroup:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = n = 10
        self.k = k = 3
        self.backend = NumpyNumericsBackend()
        self.product_manifold = UnitaryGroup(n, k=k, backend=self.backend)
        self.unitary_group = UnitaryGroup(n, backend=self.backend)
        self.unitary_group_polar = UnitaryGroup(
            n, retraction="polar", backend=self.backend
        )
        self.manifold = self.unitary_group

    def test_random_point(self):
        point = self.unitary_group.random_point()
        assert point.shape == (self.n, self.n)
        assert (point.imag > 0).any()
        self.backend.assert_allclose(
            point.T.conj() @ point, self.backend.eye(self.n)
        )

    def test_random_point_product(self):
        point = self.product_manifold.random_point()
        assert point.shape == (self.k, self.n, self.n)
        assert (point.imag > 0).any()
        self.backend.assert_allclose(
            self.backend.conjugate_transpose(point) @ point,
            self.backend.multieye(self.k, self.n),
        )

    def test_random_tangent_vector(self):
        point = self.unitary_group.random_point()
        tangent_vector = self.unitary_group.random_tangent_vector(point)
        assert (tangent_vector.imag > 0).any()
        self.backend.assert_allclose(tangent_vector, -tangent_vector.T.conj())

    def test_random_tangent_vector_product(self):
        point = self.product_manifold.random_point()
        tangent_vector = self.product_manifold.random_tangent_vector(point)
        assert (tangent_vector.imag > 0).any()
        self.backend.assert_allclose(
            tangent_vector, -self.backend.conjugate_transpose(tangent_vector)
        )

    @pytest.mark.parametrize(
        "manifold_attribute", ["unitary_group", "unitary_group_polar"]
    )
    def test_retraction(self, manifold_attribute):
        manifold = getattr(self, manifold_attribute)

        # Test that the result is on the manifold.
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        xretru = manifold.retraction(x, u)

        assert (xretru.imag > 0).any()
        self.backend.assert_allclose(
            xretru.T.conj() @ xretru, self.backend.eye(self.n)
        )

    def test_exp_log_inverse(self):
        s = self.unitary_group
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        self.backend.assert_allclose(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.unitary_group
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        self.backend.assert_allclose(U, Ulogexp)
