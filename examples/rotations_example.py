import numpy as np

import pymanopt
from pymanopt.manifolds import SpecialOrthogonalGroup
from pymanopt.solvers import TrustRegions


def main():
    n = 3
    m = 10
    k = 10

    A = np.random.randn(k, n, m)
    B = np.random.randn(k, n, m)

    if k == 1:
        A = A.reshape(n, m)
        B = B.reshape(n, m)
        ABt = A.dot(B.T)
    else:
        ABt = np.array([Ak.dot(Bk.T) for Ak, Bk in zip(A, B)])

    manifold = SpecialOrthogonalGroup(n, k)

    @pymanopt.function.Callable
    def cost(X):
        return -np.tensordot(X, ABt, axes=X.ndim)

    @pymanopt.function.Callable
    def egrad(X):
        return -ABt

    @pymanopt.function.Callable
    def ehess(X, S):
        return manifold.zerovec(X)

    problem = pymanopt.Problem(manifold, cost, egrad=egrad, ehess=ehess)

    solver = TrustRegions()

    X = solver.solve(problem)

    def sol(ABt):
        # Compare with the known optimal solution\n",
        U, S, Vt = np.linalg.svd(ABt)
        UVt = np.dot(U, Vt)
        # The determinant of UVt is either 1 or -1, in theory\n",
        if abs(1.0 - np.linalg.det(UVt)) < 1e-10:
            Xopt = UVt
        elif abs(-1.0 - np.linalg.det(UVt)) < 1e-10:
            # UVt is in O(n) but not SO(n). This is easily corrected for:\n",
            J = np.diag(np.append(np.ones(n - 1), -1))
            Xopt = np.dot(np.dot(U, J), Vt)
        else:
            raise RuntimeError('Should never happen')
        return Xopt

    if k == 1:
        Xopt = sol(ABt)
    else:
        Xopt = np.array([sol(ABtk) for ABtk in ABt])

    print('This should be small: {error}\n'.format(
        error=np.linalg.norm(Xopt - X)))


if __name__ == "__main__":
    main()
