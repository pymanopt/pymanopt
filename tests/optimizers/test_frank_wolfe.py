import autograd.numpy as np
import pytest
import scipy
import pymanopt
from pymanopt.optimizers import FrankWolfe
from pymanopt.manifolds import SymmetricPositiveDefinite

class TestFrankWolfe:
    @pytest.fixture(autouse=True)
    def setup(self):
        n = 3
        N = 50
        matrices = [np.random.normal(size=(n, n)) for i in range(N)]
        matrices = [
            matrix * matrix.T for matrix in matrices
        ]
        self.manifold = manifold = SymmetricPositiveDefinite(n)

        @pymanopt.function.autograd(manifold)
        def cost(X):
            Xinvsqrt = np.linalg.inv(scipy.linalg.sqrtm(X))
            return sum(
                np.linalg.norm(scipy.linalg.logm(Xinvsqrt @ matrix @ Xinvsqrt)) for matrix in matrices
            )
        
        self.problem = pymanopt.Problem(manifold, cost)
        L = scipy.stats.mean(matrices, axis = 0)
        U = scipy.statse.hmean(matrices, axis = 0)
        optimizer = FrankWolfe()
        self.result = optimizer.run(self.problem, L, U)


