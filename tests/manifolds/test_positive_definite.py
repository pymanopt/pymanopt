import pytest

from pymanopt.manifolds import (
    HermitianPositiveDefinite,
    SpecialHermitianPositiveDefinite,
    SymmetricPositiveDefinite,
)
from pymanopt.numerics.core import NumericsBackend


def geodesic(point_a, point_b, alpha: float, backend: NumericsBackend):
    if alpha < 0 or 1 < alpha:
        raise ValueError("Exponent must be in [0,1]")
    c = backend.linalg_cholesky(point_a)
    c_inv = backend.linalg_inv(c)
    log_cbc = backend.linalg_logm(
        c_inv @ point_b @ backend.conjugate_transpose(c_inv),
        positive_definite=True,
    )
    powm = backend.linalg_expm(alpha * log_cbc, symmetric=False)
    return c @ powm @ backend.conjugate_transpose(c)


@pytest.fixture(params=[1, 3])
def product_dimension(request) -> int:
    return request.param


class TestMultiSymmetricPositiveDefiniteManifold:
    @pytest.fixture(autouse=True)
    def setup(
        self, real_numerics_backend: NumericsBackend, product_dimension: int
    ):
        self.n = n = 10
        self.k = k = product_dimension
        self.point_shape = (k, n, n) if k > 1 else (n, n)
        self.backend = real_numerics_backend
        self.manifold = SymmetricPositiveDefinite(n, k=k, backend=self.backend)

    def test_dim(self):
        assert self.manifold.dim == self.k * self.n * (self.n + 1) // 2

    def test_typical_dist(self):
        manifold = self.manifold
        self.backend.assert_allclose(
            manifold.typical_dist, self.backend.sqrt(manifold.dim)
        )

    def test_dist(self):
        bk = self.backend
        manifold = self.manifold
        x = manifold.random_point()
        y = manifold.random_point()
        z = manifold.random_point()

        # Test separability
        bk.assert_allclose(manifold.dist(x, x), 0.0, rtol=1e-5, atol=1e-5)

        # Test symmetry
        bk.assert_allclose(manifold.dist(x, y), manifold.dist(y, x))

        # Test triangle inequality
        assert manifold.dist(x, y) <= manifold.dist(x, z) + manifold.dist(z, y)

        # Test exponential metric increasing property
        # (see equation (6.8) in [Bha2007]).
        logx, logy = bk.linalg_logm(x), bk.linalg_logm(y)
        assert manifold.dist(x, y) >= bk.linalg_norm(logx - logy)

        # check that dist is consistent with log
        bk.assert_allclose(
            manifold.dist(x, y),
            manifold.norm(x, manifold.log(x, y)),
            atol=1e-2,
        )

        # Test invariance under inversion
        bk.assert_allclose(
            manifold.dist(x, y),
            manifold.dist(bk.linalg_inv(y), bk.linalg_inv(x)),
        )

        # Test congruence-invariance (see equation (6.5) in [Bha2007]).
        a = bk.random_normal(size=(self.n, self.n))  # must be invertible
        assert bk.linalg_det(a) != 0.0
        # axa = a @ x @ bk.conjugate_transpose(a)
        # aya = a @ y @ bk.conjugate_transpose(a)

        # bk.assert_allclose(
        #     manifold.dist(x, y),
        #     manifold.dist(
        #         bk.conjugate_transpose(a) @ x @ a,
        #         bk.conjugate_transpose(a) @ y @ a,
        #     ),
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

        # Test proportionality (see equation (6.12) in [Bha2007]).
        alpha = bk.to_real_backend().random_uniform()
        bk.assert_allclose(
            manifold.dist(x, geodesic(x, y, alpha, bk)),
            alpha * manifold.dist(x, y),
            rtol=1e-5,
            atol=1e-5,
        )

    def test_inner_product(self):
        bk = self.backend
        manifold = self.manifold
        x = manifold.random_point()
        a, b = bk.herm(bk.random_normal(size=(2,) + self.point_shape))
        bk.assert_allclose(
            bk.tensordot(bk.conjugate(a), b, axes=bk.ndim(a)),
            manifold.inner_product(x, x @ a, x @ b),
        )

    def test_projection(self):
        manifold = self.manifold
        x = manifold.random_point()
        a = self.backend.random_normal(size=self.point_shape)
        self.backend.assert_allclose(
            manifold.projection(x, a), self.backend.herm(a)
        )

    def test_euclidean_to_riemannian_gradient(self):
        manifold = self.manifold
        x = manifold.random_point()
        u = self.backend.random_normal(size=(self.k, self.n, self.n))
        self.backend.assert_allclose(
            manifold.euclidean_to_riemannian_gradient(x, u),
            x @ self.backend.herm(u) @ x,
        )

    def test_euclidean_to_riemannian_hessian(self):
        # Use manopt's slow method
        manifold = self.manifold
        n = self.n
        k = self.k
        x = manifold.random_point()
        egrad, ehess = self.backend.random_normal(size=(2, k, n, n))
        u = manifold.random_tangent_vector(x)

        Hess = x @ self.backend.herm(ehess) @ x + 2 * self.backend.herm(
            u @ self.backend.herm(egrad) @ x
        )

        # Correction factor for the non-constant metric
        Hess = Hess - self.backend.herm(u @ self.backend.herm(egrad) @ x)
        self.backend.assert_allclose(
            Hess, manifold.euclidean_to_riemannian_hessian(x, egrad, ehess, u)
        )

    def test_norm(self):
        manifold = self.manifold
        x = manifold.random_point()
        Id = (
            self.backend.eye(self.n)
            if self.k == 1
            else self.backend.multieye(self.k, self.n)
        )
        self.backend.assert_allclose(
            manifold.norm(Id, x), self.backend.linalg_norm(x)
        )

    def test_random_point(self):
        manifold = self.manifold
        bk = self.backend
        x = manifold.random_point()
        # Check shape
        assert x.shape == self.point_shape
        # Check symmetry
        bk.assert_allclose(x, bk.conjugate_transpose(x))
        # Check positivity of eigenvalues
        w = bk.linalg_eigvalsh(x)
        assert bk.isrealobj(w)
        assert bk.all(w > 0.0)

    def test_random_tangent_vector(self):
        # Just test that random_tangent_vector returns an element of the tangent space
        # with norm 1 and that two random_tangent_vectors are different.
        manifold = self.manifold
        bk = self.backend
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        assert u.shape == self.point_shape
        bk.assert_allclose(u, bk.conjugate_transpose(u))
        bk.assert_allclose(1.0, manifold.norm(x, u))
        v = manifold.random_tangent_vector(x)
        assert bk.linalg_norm(u - v) > 1e-3

    def test_transport(self):
        manifold = self.manifold
        x = manifold.random_point()
        y = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        u_transp = manifold.transport(x, y, u)
        u_transp_proj = manifold.projection(y, u_transp)
        self.backend.assert_allclose(u_transp, u_transp_proj)

    def test_exp(self):
        # Test against manopt implementation, test that for small vectors
        # exp(x, u) = x + u.
        manifold = self.manifold
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        e = self.backend.linalg_expm(self.backend.linalg_solve(x, u))

        self.backend.assert_allclose(x @ e, manifold.exp(x, u))
        u = u * 1e-6
        self.backend.assert_allclose(manifold.exp(x, u), x + u)

    def test_retraction(self):
        # Check that result is on manifold and for small vectors
        # retr(x, u) = x + u.
        manifold = self.manifold
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        y = manifold.retraction(x, u)

        assert y.shape == self.point_shape
        # Check symmetry
        self.backend.assert_allclose(y, self.backend.herm(y))

        # Check positivity of eigenvalues
        w = self.backend.linalg_eigvalsh(y)
        assert self.backend.all(w > 0.0)

        u = u * 1e-6
        self.backend.assert_allclose(manifold.retraction(x, u), x + u)

    def test_exp_log_inverse(self):
        manifold = self.manifold
        x = manifold.random_point()
        y = manifold.random_point()
        u = manifold.log(x, y)
        self.backend.assert_allclose(manifold.exp(x, u), y)

    def test_log_exp_inverse(self):
        manifold = self.manifold
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        y = manifold.exp(x, u)
        self.backend.assert_allclose(
            manifold.log(x, y), u, rtol=1e-5, atol=1e-5
        )


class TestMultiHermitianPositiveDefiniteManifold(
    TestMultiSymmetricPositiveDefiniteManifold
):
    @pytest.fixture(autouse=True)
    def setup(
        self, complex_numerics_backend: NumericsBackend, product_dimension: int
    ):
        self.n = n = 10
        self.k = k = product_dimension
        self.point_shape = (k, n, n) if k > 1 else (n, n)
        self.backend = complex_numerics_backend
        self.manifold = HermitianPositiveDefinite(n, k=k, backend=self.backend)

    def test_dim(self):
        assert self.manifold.dim == self.k * self.n * (self.n + 1)


class TestMultiSpecialHermitianPositiveDefiniteManifold(
    TestMultiHermitianPositiveDefiniteManifold
):
    @pytest.fixture(autouse=True)
    def setup(
        self, complex_numerics_backend: NumericsBackend, product_dimension: int
    ):
        self.n = n = 10
        self.k = k = product_dimension
        self.point_shape = (k, n, n) if k > 1 else (n, n)
        self.det_shape = (k,) if k > 1 else ()
        self.backend = complex_numerics_backend
        self.manifold = SpecialHermitianPositiveDefinite(
            n, k=k, backend=self.backend
        )

    def test_dim(self):
        manifold = self.manifold
        n = self.n
        k = self.k
        self.backend.assert_allclose(manifold.dim, k * (n * (n + 1) - 1))

    def test_random_point(self):
        # Just test that rand returns a point on the manifold and two
        # different matrices generated by rand aren't too close together
        manifold = self.manifold
        bk = self.backend
        x = manifold.random_point()
        # Check shape.
        assert x.shape == self.point_shape
        # Check symmetry.
        bk.assert_allclose(x, bk.conjugate_transpose(x))
        # Check positivity of eigenvalues.
        w = bk.linalg_eigvalsh(x)
        assert bk.all(w > 0.0)
        # Check unit determinant.
        bk.assert_allclose(bk.linalg_det(x), bk.ones(self.det_shape))
        # Check randomness.
        y = manifold.random_point()
        assert bk.linalg_norm(x - y) > 1e-3

    def test_random_tangent_vector(self):
        # Just test that randvec returns an element of the tangent space
        # with norm 1 and that two randvecs are different.
        manifold = self.manifold
        bk = self.backend
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        bk.assert_allclose(bk.conjugate_transpose(u), u)
        #
        bk.assert_allclose(
            bk.trace(bk.linalg_solve(x, u)),
            bk.zeros(self.det_shape),
            atol=1e-7,
        )
        # Check unit norm.
        bk.assert_allclose(1.0, manifold.norm(x, u))

        v = manifold.random_tangent_vector(x)
        assert bk.linalg_norm(u - v) > 1e-3

    def test_projection(self):
        manifold = self.manifold
        bk = self.backend
        x = manifold.random_point()
        a = bk.random_normal(size=self.point_shape)
        p = manifold.projection(x, a)
        # Check shape
        assert p.shape == self.point_shape
        # Check hermitian symmetry.
        bk.assert_allclose(p, bk.conjugate_transpose(p))
        #
        bk.assert_allclose(
            bk.trace(bk.linalg_solve(x, p)),
            bk.zeros(self.det_shape),
            atol=1e-7,
        )
        # Check invariance of projection
        bk.assert_allclose(p, manifold.projection(x, p))

    def test_euclidean_to_riemannian_gradient(self):
        manifold = self.manifold
        x = manifold.random_point()
        u = self.backend.random_normal(size=(self.k, self.n, self.n))
        self.backend.assert_allclose(
            manifold.euclidean_to_riemannian_gradient(x, u),
            manifold.projection(x, x @ u @ x),
        )

    def test_euclidean_to_riemannian_hessian(self):
        pass

    def test_exp(self):
        # Test against manopt implementation, test that for small vectors
        # exp(x, u) = x + u.
        manifold = self.manifold
        bk = self.backend
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        e = manifold.exp(x, u)

        # Check symmetry
        bk.assert_allclose(e, bk.herm(e))

        # Check positivity of eigenvalues
        w = bk.linalg_eigvalsh(e)
        assert bk.all(w > 0.0)

        # Check unit determinant
        d = bk.linalg_det(e)
        bk.assert_allclose(d, bk.ones(self.det_shape))

        u = u * 1e-6
        bk.assert_allclose(manifold.exp(x, u), x + u)

    def test_retraction(self):
        # Check that result is on manifoldifold and for small vectors
        # retraction(x, u) = x + u.
        manifold = self.manifold
        bk = self.backend
        x = manifold.random_point()
        u = manifold.random_tangent_vector(x)
        y = manifold.retraction(x, u)

        assert y.shape == self.point_shape
        # Check symmetry
        bk.assert_allclose(y, bk.conjugate_transpose(y))

        # Check positivity of eigenvalues
        w = bk.linalg_eigvalsh(y)
        assert bk.all(w > 0.0)

        # Check unit determinant
        d = bk.linalg_det(y)
        bk.assert_allclose(d, bk.ones(self.det_shape))

        u = u * 1e-6
        bk.assert_allclose(manifold.retraction(x, u), x + u)
