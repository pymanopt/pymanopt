import pytest

from pymanopt.backends import Backend
from pymanopt.manifolds import SpecialOrthogonalGroup, UnitaryGroup


class TestSpecialOrthogonalGroup:
    @pytest.fixture(autouse=True)
    def setup(self, real_backend: Backend):
        self.n = n = 10
        self.k = k = 3
        self.backend = real_backend
        self.so_product = SpecialOrthogonalGroup(n, k=k, backend=self.backend)
        self.so = SpecialOrthogonalGroup(n, backend=self.backend)
        self.so_polar = SpecialOrthogonalGroup(
            n, retraction="polar", backend=self.backend
        )
        self.manifold = self.so

    def test_random_point(self):
        bk = self.backend
        point = self.so.random_point()
        assert point.shape == (self.n, self.n)
        bk.assert_allclose(bk.transpose(point) @ point, bk.eye(self.n))
        bk.assert_allclose(bk.linalg_det(point), 1.0)

    def test_random_point_product(self):
        bk = self.backend
        point = self.so_product.random_point()
        assert point.shape == (self.k, self.n, self.n)
        bk.assert_allclose(
            bk.transpose(point) @ point,
            bk.multieye(self.k, self.n),
        )
        bk.assert_allclose(bk.linalg_det(point), 1.0)

    def test_random_tangent_vector(self):
        point = self.so.random_point()
        tangent_vector = self.so.random_tangent_vector(point)
        self.backend.assert_allclose(
            tangent_vector, -self.backend.transpose(tangent_vector)
        )

    def test_random_tangent_vector_product(self):
        point = self.so_product.random_point()
        tangent_vector = self.so_product.random_tangent_vector(point)
        self.backend.assert_allclose(
            tangent_vector, -self.backend.transpose(tangent_vector)
        )

    @pytest.mark.parametrize("manifold_attribute", ["so", "so_polar"])
    def test_retraction(self, manifold_attribute):
        bk = self.backend
        manifold: SpecialOrthogonalGroup = getattr(self, manifold_attribute)

        # Test that the result is on the manifold.
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        xretru = manifold.retraction(x, u)

        # Check the retraction is a point on the manifold, i.e. it is
        # orthogonal and has det 1
        bk.assert_allclose(
            bk.transpose(xretru) @ xretru, bk.eye(self.n), atol=5e-6
        )
        bk.assert_allclose(bk.linalg_det(xretru), 1.0, atol=1e-5)

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
    def setup(self, complex_backend: Backend):
        self.n = n = 10
        self.k = k = 3
        self.backend = complex_backend
        self.product_manifold = UnitaryGroup(n, k=k, backend=self.backend)
        self.unitary_group = UnitaryGroup(n, backend=self.backend)
        self.unitary_group_polar = UnitaryGroup(
            n, retraction="polar", backend=self.backend
        )
        self.manifold = self.unitary_group

    def test_random_point(self):
        bk = self.backend
        point = self.unitary_group.random_point()
        assert point.shape == (self.n, self.n)
        assert bk.any(bk.imag(point) > 0)
        bk.assert_allclose(
            bk.conjugate_transpose(point) @ point, bk.eye(self.n)
        )

    def test_random_point_product(self):
        bk = self.backend
        point = self.product_manifold.random_point()
        assert point.shape == (self.k, self.n, self.n)
        assert bk.any(bk.imag(point) > 0.0)
        bk.assert_allclose(
            bk.conjugate_transpose(point) @ point,
            bk.multieye(self.k, self.n),
        )

    def test_random_tangent_vector(self):
        bk = self.backend
        point = self.unitary_group.random_point()
        tangent_vector = self.unitary_group.random_tangent_vector(point)
        assert bk.any(bk.imag(tangent_vector) > 0.0)
        bk.assert_allclose(
            tangent_vector, -bk.conjugate_transpose(tangent_vector)
        )

    def test_random_tangent_vector_product(self):
        bk = self.backend
        point = self.product_manifold.random_point()
        tangent_vector = self.product_manifold.random_tangent_vector(point)
        assert bk.any(bk.imag(tangent_vector) > 0.0)
        bk.assert_allclose(
            tangent_vector, -bk.conjugate_transpose(tangent_vector)
        )

    @pytest.mark.parametrize(
        "manifold_attribute", ["unitary_group", "unitary_group_polar"]
    )
    def test_retraction(self, manifold_attribute):
        bk = self.backend
        manifold = getattr(self, manifold_attribute)

        # Test that the result is on the manifold.
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        xretru = manifold.retraction(x, u)

        assert bk.any(bk.imag(xretru) > 0.0)
        bk.assert_allclose(
            bk.conjugate_transpose(xretru) @ xretru, bk.eye(self.n)
        )

    def test_exp_log_inverse(self):
        bk = self.backend
        s = self.unitary_group
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        bk.assert_allclose(Y, Yexplog)

    def test_log_exp_inverse(self):
        bk = self.backend
        s = self.unitary_group
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        bk.assert_allclose(U, Ulogexp)
