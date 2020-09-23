import numpy as np

from pymanopt.manifolds.manifold import Manifold


class PoincareBall(Manifold):
    """
    Factory class for the Poincare ball model of hyperbolic geometry.
    An instance represents the Cartesian product of n (defaults to 1)
    Poincare balls of dimension k.
    Elements are represented as matrices of size k x n, or arrays of
    size k if n = 1.

    The Poincare ball is embedded in R^k and is a Riemannian manifold,
    but it is not an embedded Riemannian submanifold.
    At every point x, the tangent space at x is R^k since the manifold
    is open.

    The metric is conformal to the Euclidean one (angles are preserved),
    and it is given at every point x by
        <u, v>_x = lambda_x^2 <u, v>,
    where lambda_x = 2 / (1 - norm{x}^2) is the conformal factor.
    This induces the following distance between any two points x and y:
        dist(x, y) = acosh(
            1 + 2 norm{x - y}^2 / (1 - norm{x}^2) / (1 - norm{y}^2)
        ).
    """

    def __init__(self, k: int, n: int = 1):
        self._k = k
        self._n = n

        if k < 1:
            raise ValueError(f"Need k >= 1. Value supplied was {k}")
        if n < 1:
            raise ValueError(f"Need n >= 1. Value supplied was {n}")

        if n == 1:
            name = f"Poincare ball B({k})"
        elif n >= 2:
            name = f"Product Poincare ball B({k})^{n}"

        dimension = k * n
        super().__init__(name, dimension)

    @property
    def typicaldist(self):
        return self.dim / 8

    # The metric in the Poincare ball is conformal to the Euclidean one.
    def conformal_factor(self, X):
        return 2 / (1 - np.sum(X * X, axis=0))

    def inner(self, X, G, H):
        factors = np.square(self.conformal_factor(X))
        return np.sum(G * H * factors)

    # Identity map since the tangent space is the ambient space.
    def proj(self, X, G):
        return G

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    # Generates points sampled uniformly at random in the unit ball.
    # In high dimension (large k), sampled points are very likely to
    # be close to the boundary because of the curse of dimensionality.
    def rand(self):
        if self._n == 1:
            N = np.random.randn(self._k)
        else:
            N = np.random.randn(self._k, self._n)
        norms = np.linalg.norm(N, axis=0)
        radiuses = np.random.rand(self._n) ** (1.0 / self._k)
        return radiuses * N / norms

    def randvec(self, X):
        return np.random.randn(*np.shape(X))

    def zerovec(self, X):
        return np.zeros(np.shape(X))

    # Geodesic distance.
    def dist(self, X, Y):
        norms2_X = np.sum(X * X, axis=0)
        norms2_Y = np.sum(Y * Y, axis=0)
        difference = X - Y
        norms2_difference = np.sum(difference * difference, axis=0)

        columns_dist = np.arccosh(
            1 + 2 * norms2_difference / ((1 - norms2_X) * (1 - norms2_Y))
        )
        return np.sqrt(np.sum(np.square(columns_dist)))

    # The hyperbolic metric tensor is conformal to the Euclidean one,
    # so the Euclidean gradient is simply rescaled.
    def egrad2rgrad(self, X, G):
        factors = np.square(1 / self.conformal_factor(X))
        return G * factors

    # Derived from the Koszul formula.
    def ehess2rhess(self, X, G, H, U):
        lambda_x = self.conformal_factor(X)
        return (
            np.sum(G * X, axis=0) * U
            - np.sum(X * U, axis=0) * G
            - np.sum(G * U, axis=0) * X
            + H / lambda_x
        ) / lambda_x

    # Exponential map is cheap so use it as a retraction.
    def retr(self, X, G):
        return self.exp(X, G)

    # Special non-associative and non-commutative operation
    # which is closed in the Poincare ball.
    # Performed column-wise here.
    def mobius_addition(self, X, Y):
        scalar_product = np.sum(X * Y, axis=0)
        norm2X = np.sum(X * X, axis=0)
        norm2Y = np.sum(Y * Y, axis=0)

        return (X * (1 + 2 * scalar_product + norm2Y) + Y * (1 - norm2X)) / (
            1 + 2 * scalar_product + norm2X * norm2Y
        )

    def exp(self, X, U):
        norm_U = np.linalg.norm(U, axis=0)
        # Handle the case where U is null.
        W = U * np.divide(
            np.tanh(norm_U / (1 - np.sum(X * X, axis=0))),
            norm_U,
            out=np.zeros_like(U),
            where=norm_U != 0,
        )
        return self.mobius_addition(X, W)

    def log(self, X, Y):
        W = self.mobius_addition(-X, Y)
        norm_W = np.linalg.norm(W, axis=0)
        return (1 - np.sum(X * X, axis=0)) * np.arctanh(norm_W) * W / norm_W

    # I don't have a nice expression for this.
    # To be completed in the future.
    def transp(self, X1, X2, G):
        raise NotImplementedError

    def pairmean(self, X, Y):
        return self.exp(X, self.log(X, Y) / 2)
