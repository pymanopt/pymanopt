import autograd.numpy as np
import numpy.testing as np_testing
import pytest

import pymanopt
from pymanopt.optimizers import ConjugateGradient


class TestConjugateGradient:
    @pytest.fixture(autouse=True)
    def setup(self):
        n = 32
        matrix = np.random.normal(size=(n, n))
        matrix = 0.5 * (matrix + matrix.T)

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        self.dominant_eigenvector = eigenvectors[:, np.argmax(eigenvalues)]

        self.manifold = manifold = pymanopt.manifolds.Sphere(n)

        @pymanopt.function.autograd(manifold)
        def cost(point):
            return -point.T @ matrix @ point

        self.problem = pymanopt.Problem(manifold, cost)

    @pytest.mark.parametrize(
        "beta_rule",
        [
            "FletcherReeves",
            "HagerZhang",
            "HestenesStiefel",
            "PolakRibiere",
            "LiuStorey",
        ],
    )
    def test_beta_rules(self, beta_rule):
        optimizer = ConjugateGradient(beta_rule=beta_rule, verbosity=0)
        result = optimizer.run(self.problem)
        estimated_dominant_eigenvector = result.point
        if np.sign(self.dominant_eigenvector[0]) != np.sign(
            estimated_dominant_eigenvector[0]
        ):
            estimated_dominant_eigenvector = -estimated_dominant_eigenvector
        np_testing.assert_allclose(
            self.dominant_eigenvector,
            estimated_dominant_eigenvector,
            atol=1e-6,
        )

    def test_beta_invalid_rule(self):
        with pytest.raises(ValueError):
            ConjugateGradient(beta_rule="SomeUnknownBetaRule")

    def test_complex_cost_problem(self):
        # Solve the dominant invariant complex subspace problem.
        num_rows = 32
        subspace_dimension = 3
        matrix = np.random.normal(
            size=(num_rows, num_rows)
        ) + 1j * np.random.normal(size=(num_rows, num_rows))
        matrix = 0.5 * (matrix + matrix.T.conj())

        manifold = pymanopt.manifolds.ComplexGrassmann(
            num_rows, subspace_dimension
        )

        @pymanopt.function.autograd(manifold)
        def cost(X):
            return -np.real(np.trace(np.conj(X.T) @ matrix @ X))

        problem = pymanopt.Problem(manifold, cost)
        optimizer = ConjugateGradient(verbosity=0)
        estimated_spanning_set = optimizer.run(problem).point

        # True solution.
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        column_indices = np.argsort(eigenvalues)[-subspace_dimension:]
        spanning_set = eigenvectors[:, column_indices]
        np_testing.assert_allclose(
            manifold.dist(spanning_set, estimated_spanning_set), 0, atol=1e-5
        )
