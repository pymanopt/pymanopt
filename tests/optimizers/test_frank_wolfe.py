import autograd.numpy as np
import pytest
import scipy
import pymanopt
from pymanopt.optimizers import FrankWolfe
from pymanopt.manifolds import SymmetricPositiveDefinite

n = 100
N = 50
matrices = [np.random.uniform(size=(n, n)) for i in range(N)]
matricespd = np.array([
    (matrix @ matrix.T)for matrix in matrices
])
matricesinv = np.array([
    np.linalg.inv(matrix) for matrix in matricespd
])
manifold = manifold = SymmetricPositiveDefinite(n)
@pymanopt.function.autograd(manifold)
def cost(X):
    # print(type(X))
    X = np.array(X)
    # print(type(X))
    # evalues, evectors = np.linalg.eig()
    Xinvsqrt = np.array(np.linalg.inv(scipy.linalg.sqrtm(X)))
    # print(type(Xinvsqrt))
    log_matrix_array = np.array([np.linalg.norm(scipy.linalg.logm(Xinvsqrt @ matrix @ Xinvsqrt), ord='fro')**2 for matrix in matricespd])
    return 1/N * sum(
        # np.linalg.norm(scipy.linalg.logm(Xinvsqrt @ matrix @ Xinvsqrt)) for matrix in matricespd
        log_matrix_array
    )
@pymanopt.function.autograd(manifold)
def rieman_grad(X, matricesinv = matricesinv):
      print(X)
      Xinv = np.linalg.inv(X)
      return 1/N * Xinv @ np.sum(
        scipy.linalg.logm(X @ matrix) for matrix in matricesinv
    )
problem = pymanopt.Problem(manifold, cost, riemannian_gradient=rieman_grad)
U = np.mean(matricespd, axis = 0)
U = U
print(matricespd)
L = np.linalg.inv(np.sum(1/N * matricesinv, axis = 0))
L = L
print(L)
print(U)
optimizer = FrankWolfe(max_iterations = 20, log_verbosity = 1)
initial_point = U
print(np.linalg.cholesky(U))
print(np.linalg.cholesky(L))
result = optimizer.run(problem, L, U, initial_point = initial_point)
print(result)
