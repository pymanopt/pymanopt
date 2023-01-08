import autograd.numpy as np
import numpy.testing as np_testing
import pytest

from pymanopt.manifolds import SpecialOrthogonalGroup, UnitaryGroup
from pymanopt.tools.multi import multieye, multihconj, multitransp


class TestSpecialOrthogonalGroup:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = n = 10
        self.k = k = 3
        self.so_product = SpecialOrthogonalGroup(n, k=k)
        self.so = SpecialOrthogonalGroup(n)
        self.so_polar = SpecialOrthogonalGroup(n, retraction="polar")
        self.manifold = self.so

    def test_random_point(self):
        point = self.so.random_point()
        assert point.shape == (self.n, self.n)
        np_testing.assert_almost_equal(point.T @ point - point @ point.T, 0)
        np_testing.assert_almost_equal(point.T @ point, np.eye(self.n))
        assert np.allclose(np.linalg.det(point), 1)

    def test_random_point_product(self):
        point = self.so_product.random_point()
        assert point.shape == (self.k, self.n, self.n)
        np_testing.assert_almost_equal(
            multitransp(point) @ point - point @ multitransp(point),
            0,
        )
        np_testing.assert_almost_equal(
            multitransp(point) @ point, multieye(self.k, self.n)
        )
        assert np.allclose(np.linalg.det(point), 1)

    def test_random_tangent_vector(self):
        point = self.so.random_point()
        tangent_vector = self.so.random_tangent_vector(point)
        np_testing.assert_almost_equal(tangent_vector, -tangent_vector.T)

    def test_random_tangent_vector_product(self):
        point = self.so_product.random_point()
        tangent_vector = self.so_product.random_tangent_vector(point)
        np_testing.assert_almost_equal(
            tangent_vector, -multitransp(tangent_vector)
        )

    @pytest.mark.parametrize("manifold_attribute", ["so", "so_polar"])
    def test_retraction(self, manifold_attribute):
        manifold = getattr(self, manifold_attribute)

        # Test that the result is on the manifold.
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        xretru = manifold.retraction(x, u)

        np_testing.assert_almost_equal(
            xretru.T @ xretru - xretru @ xretru.T, 0
        )
        np_testing.assert_almost_equal(
            xretru.T @ xretru, np.eye(xretru.shape[-1])
        )
        np_testing.assert_allclose(np.linalg.det(xretru), 1)

    def test_exp_log_inverse(self):
        s = self.so
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        np_testing.assert_array_almost_equal(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.so
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U, Ulogexp)


class TestUnitaryGroup:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.n = n = 10
        self.k = k = 3
        self.product_manifold = UnitaryGroup(n, k=k)
        self.unitary_group = UnitaryGroup(n)
        self.unitary_group_polar = UnitaryGroup(n, retraction="polar")
        self.manifold = self.unitary_group

    def test_random_point(self):
        point = self.unitary_group.random_point()
        assert point.shape == (self.n, self.n)
        assert (point.imag > 0).any()
        np_testing.assert_almost_equal(point.T.conj() @ point, np.eye(self.n))

    def test_random_point_product(self):
        point = self.product_manifold.random_point()
        assert point.shape == (self.k, self.n, self.n)
        assert (point.imag > 0).any()
        np_testing.assert_almost_equal(
            multihconj(point) @ point, multieye(self.k, self.n)
        )

    def test_random_tangent_vector(self):
        point = self.unitary_group.random_point()
        tangent_vector = self.unitary_group.random_tangent_vector(point)
        assert (tangent_vector.imag > 0).any()
        np_testing.assert_almost_equal(
            tangent_vector, -tangent_vector.T.conj()
        )

    def test_random_tangent_vector_product(self):
        point = self.product_manifold.random_point()
        tangent_vector = self.product_manifold.random_tangent_vector(point)
        assert (tangent_vector.imag > 0).any()
        np_testing.assert_almost_equal(
            tangent_vector, -multihconj(tangent_vector)
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
        np_testing.assert_almost_equal(
            xretru.T.conj() @ xretru, np.eye(self.n)
        )

    def test_exp_log_inverse(self):
        s = self.unitary_group
        X = s.random_point()
        Y = s.random_point()
        Yexplog = s.exp(X, s.log(X, Y))
        np_testing.assert_array_almost_equal(Y, Yexplog)

    def test_log_exp_inverse(self):
        s = self.unitary_group
        X = s.random_point()
        U = s.random_tangent_vector(X)
        Ulogexp = s.log(X, s.exp(X, U))
        np_testing.assert_array_almost_equal(U, Ulogexp)
