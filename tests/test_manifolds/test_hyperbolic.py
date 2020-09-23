import autograd.numpy as np
from numpy import linalg as la, testing as np_testing

from pymanopt.manifolds import PoincareBall
from .._test import TestCase


class TestSinglePoincareBallManifold(TestCase):
    def setUp(self):
        self.k = 50
        self.man = PoincareBall(self.k)

    def test_dim(self):
        assert self.man.dim == self.k

    # def test_typicaldist(self):
    #     pass

    def test_conformal_factor(self):
        x = self.man.rand() / 2
        np_testing.assert_allclose(
            1 - 2 / self.man.conformal_factor(x), la.norm(x) ** 2
        )

    def test_inner(self):
        x = self.man.rand() / 2
        u = self.man.randvec(x)
        v = self.man.randvec(x)
        np_testing.assert_allclose(
            (2 / (1 - la.norm(x) ** 2)) ** 2 * np.inner(u, v),
            self.man.inner(x, u, v),
        )

        # test that angles are preserved
        x = self.man.rand() / 2
        u = self.man.randvec(x)
        v = self.man.randvec(x)
        cos_eangle = np.sum(u * v) / la.norm(u) / la.norm(v)
        cos_rangle = (
            self.man.inner(x, u, v) / self.man.norm(x, u) / self.man.norm(x, v)
        )
        np_testing.assert_allclose(cos_rangle, cos_eangle)

    def test_proj(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        np_testing.assert_allclose(u, self.man.proj(x, u))

    def test_norm(self):
        x = self.man.rand() / 2
        u = self.man.randvec(x)

        np_testing.assert_allclose(
            2 / (1 - la.norm(x) ** 2) * la.norm(u), self.man.norm(x, u)
        )

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        x = self.man.rand()
        np_testing.assert_array_less(la.norm(x), 1)
        y = self.man.rand()
        assert not np.array_equal(x, y)

    def test_randvec(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        x = self.man.rand()
        u = self.man.randvec(x)
        v = self.man.randvec(x)

        assert not np.array_equal(u, v)

    def test_zerovec(self):
        x = self.man.rand()
        u = self.man.zerovec(x)
        np_testing.assert_allclose(la.norm(u), 0)

    def test_dist(self):
        x = self.man.rand() / 2
        y = self.man.rand() / 2
        correct_dist = np.arccosh(
            1
            + 2
            * la.norm(x - y) ** 2
            / (1 - la.norm(x) ** 2)
            / (1 - la.norm(y) ** 2)
        )
        np_testing.assert_allclose(correct_dist, self.man.dist(x, y))

    # def test_egrad2rgrad(self):
    #     pass

    # def test_ehess2rhess(self):
    #     pass

    def test_retr(self):
        x = self.man.rand() / 2
        u = self.man.randvec(x)
        y = self.man.retr(x, u)
        assert la.norm(y) <= 1 + 1e-10

    def test_mobius_addition(self):
        # test if Mobius addition is closed in the Poincare ball
        x = self.man.rand() / 2
        y = self.man.rand() / 2
        z = self.man.mobius_addition(x, y)
        # The norm of z may be slightly more than one because of
        # round-off errors.
        assert la.norm(z) <= 1 + 1e-10

    def test_exp_log_inverse(self):
        x = self.man.rand() / 2
        y = self.man.rand() / 2
        explog = self.man.exp(x, self.man.log(x, y))
        np_testing.assert_allclose(y, explog)

    def test_log_exp_inverse(self):
        x = self.man.rand() / 2
        # If u is too big its exponential will have norm 1 because of
        # numerical approximations
        u = self.man.randvec(x) / self.man.dim
        logexp = self.man.log(x, self.man.exp(x, u))
        np_testing.assert_allclose(u, logexp)

    # def test_transp(self):
    #     pass

    def test_pairmean(self):
        x = self.man.rand() / 2
        y = self.man.rand() / 2
        z = self.man.pairmean(x, y)
        np_testing.assert_allclose(self.man.dist(x, z), self.man.dist(y, z))


class TestMultiplePoincareBallManifold(TestCase):
    def setUp(self):
        self.k = 50
        self.n = 20
        self.man = PoincareBall(self.k, self.n)

    def test_dim(self):
        assert self.man.dim == self.k * self.n

    # def test_typicaldist(self):
    #     pass

    def test_conformal_factor(self):
        x = self.man.rand() / 2
        np_testing.assert_allclose(
            1 - 2 / self.man.conformal_factor(x), la.norm(x, axis=0) ** 2
        )

    def test_inner(self):
        x = self.man.rand() / 2
        u = self.man.randvec(x)
        v = self.man.randvec(x)
        np_testing.assert_allclose(
            np.sum(
                (2 / (1 - la.norm(x, axis=0) ** 2)) ** 2
                * np.sum(u * v, axis=0)
            ),
            self.man.inner(x, u, v),
        )

    def test_proj(self):
        x = self.man.rand()
        u = self.man.randvec(x)
        np_testing.assert_allclose(u, self.man.proj(x, u))

    def test_norm(self):
        # Divide by 2 to avoid round-off errors.
        x = self.man.rand() / 2
        u = self.man.randvec(x)

        np_testing.assert_allclose(
            np.sum(
                (2 / (1 - la.norm(x, axis=0) ** 2)) ** 2
                * np.sum(u * u, axis=0)
            ),
            self.man.norm(x, u) ** 2,
        )

    def test_rand(self):
        # Just make sure that things generated are on the manifold and that
        # if you generate two they are not equal.
        x = self.man.rand()
        np_testing.assert_array_less(la.norm(x, axis=0), 1)
        y = self.man.rand()
        assert not np.array_equal(x, y)

    def test_randvec(self):
        # Just make sure that things generated are in the tangent space and
        # that if you generate two they are not equal.
        x = self.man.rand()
        u = self.man.randvec(x)
        v = self.man.randvec(x)

        assert not np.array_equal(u, v)

    def test_zerovec(self):
        x = self.man.rand()
        u = self.man.zerovec(x)
        np_testing.assert_allclose(la.norm(u), 0)

    def test_dist(self):
        x = self.man.rand() / 2
        y = self.man.rand() / 2
        correct_dist = np.sum(
            np.arccosh(
                1
                + 2
                * la.norm(x - y, axis=0) ** 2
                / (1 - la.norm(x, axis=0) ** 2)
                / (1 - la.norm(y, axis=0) ** 2)
            )
            ** 2
        )
        np_testing.assert_allclose(correct_dist, self.man.dist(x, y) ** 2)

    # def test_egrad2rgrad(self):
    #     pass

    # def test_ehess2rhess(self):
    #     pass

    def test_retr(self):
        x = self.man.rand() / 2
        u = self.man.randvec(x)
        y = self.man.retr(x, u)
        np_testing.assert_array_less(la.norm(y, axis=0), 1 + 1e-10)

    def test_mobius_addition(self):
        # test if Mobius addition is closed in the Poincare ball
        x = self.man.rand()
        y = self.man.rand()
        z = self.man.mobius_addition(x, y)
        # The norm of z may be slightly more than one because of
        # round-off errors.
        np_testing.assert_array_less(la.norm(z, axis=0), 1 + 1e-10)

    def test_exp_log_inverse(self):
        x = self.man.rand() / 2
        y = self.man.rand() / 2
        explog = self.man.exp(x, self.man.log(x, y))
        np_testing.assert_allclose(y, explog)

    def test_log_exp_inverse(self):
        x = self.man.rand() / 2
        # If u is too big its exponential will have norm 1 because of
        # numerical approximations
        u = self.man.randvec(x) / self.man.dim
        logexp = self.man.log(x, self.man.exp(x, u))
        np_testing.assert_allclose(u, logexp)

    # def test_transp(self):
    #     pass

    def test_pairmean(self):
        x = self.man.rand() / 2
        y = self.man.rand() / 2
        z = self.man.pairmean(x, y)
        np_testing.assert_allclose(self.man.dist(x, z), self.man.dist(y, z))
