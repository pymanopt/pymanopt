import pytest

from pymanopt.manifolds import Oblique


class TestObliqueManifold:
    @pytest.fixture(autouse=True)
    def setup(self, real_numerics_backend):
        self.m = m = 100
        self.n = n = 50
        self.backend = real_numerics_backend
        self.manifold = Oblique(m, n, backend=self.backend)

    # def test_dim(self):

    # def test_typical_dist(self):

    # def test_dist(self):

    # def test_inner_product(self):

    # def test_projection(self):

    # def test_euclidean_to_riemannian_hessian(self):

    # def test_retraction(self):

    # def test_norm(self):

    # def test_random_point(self):

    # def test_random_tangent_vector(self):

    # def test_transport(self):

    def test_exp_log_inverse(self):
        s = self.manifold
        x = s.random_point()
        y = s.random_point()
        u = s.log(x, y)
        z = s.exp(x, u)
        self.backend.assert_allclose(s.dist(y, z), 0.0, rtol=1e-5, atol=1e-2)

    def test_log_exp_inverse(self):
        s = self.manifold
        x = s.random_point()
        u = s.random_tangent_vector(x)
        y = s.exp(x, u)
        v = s.log(x, y)
        # Check that the manifold difference between the tangent vectors u and
        # v is 0
        self.backend.assert_allclose(
            s.norm(x, u - v), 0.0, rtol=1e-5, atol=1e-2
        )

    def test_pair_mean(self):
        s = self.manifold
        X = s.random_point()
        Y = s.random_point()
        Z = s.pair_mean(X, Y)
        self.backend.assert_allclose(s.dist(X, Z), s.dist(Y, Z))
