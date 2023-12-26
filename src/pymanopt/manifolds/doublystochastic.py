import numpy as np

from pymanopt.manifolds.manifold import Manifold, RetrAsExpMixin


def randomNonNegTensor(m):
    return abs(np.randn([m, m]))

def rowColDiffNorm(x):
    (m, n) = x.size()
    (em, en) = np.ones([m]), np.ones([n])
    rd = np.norm(x @ em - em)
    cd = np.norm(x.T @ en - en)
    # if verbose:
    #     print(f'row norm: {rowNorm}, col norm: {colNorm}')
    # return rowNorm <= tol and colNorm <= tol
    return rd, cd

def sinkhorn(x0, verbose=False, maxiter: int = 100, tol=1e-3):
    """ fixpoint iteration of Sinkhorn-Knopp from: https://strathprints.strath.ac.uk/19685/1/skapp.pdf
    :param x0: matrix with positive entries
    :param verbose: debug printout
    :param maxiter: max # of iterations
    :param tol: early return tolerance
    :return: doubly stochastic matrix
    """
    m, _ = x0.size()
    onesm = np.ones([m])
    r = onesm
    for i in range(maxiter):
        c = np.diag(np.reciprocal(x0.T @ r)) @ onesm
        r = np.diag(np.reciprocal(x0   @ c)) @ onesm
        xHat = np.diag(r) @ x0 @ np.diag(c)
        rn, cn = rowColDiffNorm(xHat)
        if verbose and i % 10 == 0:
            print(f'row diff: {rn}, col diff: {cn}')
        if rn <= tol and cn <= tol:
            break
    if verbose:
        print(f'done in {i} iters at tol {tol}')
    return xHat

class DoublyStochastic(Manifold):
    def __init__(self, n, name, dimension):
        self._n = n
        self._k = 1  # a single manifold
        self.onesn = np.ones([n])
        self.idn = np.eye(n)
        super().__init__(name, dimension)
    def projection(self, x, z):
        """ orthogonal projection on the tangent
        Eqn. B.11 of https://arxiv.org/pdf/1802.02628.pdf
        :param x: point on the manifold at which the tangent is computed
        :param z: point to be projected
        :return: point on the tangent
        """
        # # solve A x = b in the least squares sense
        # torch.linalg.lstsq(A, b).solution == A.pinv() @ b

        # # Eqn B.9
        alpha = np.lstsq(self.idn - x @ x.T, (z - (x @ z.T)) @ self.onesn).x
        # # Eqn B.10
        beta = z.T @ self.onesn - x.T @ alpha

        return z - np.kron(alpha @ self.onesn + self.onesn @ beta, x)

    def retraction(self, x, v):
        """ Retraction on the manifold
        Eqn 34 of https://arxiv.org/pdf/1802.02628.pdf
        :param x: point on the manifold at which the tangent is computed
        :param v: point to be retracted on the manifold (written "xi" in the paper)
        :return: point on the manifold
        """
        return sinkhorn(x * np.exp(np.div(v, x)))

    def random_point(self):
        return sinkhorn(randomNonNegTensor(self._n))

    def euclidean_to_riemannian_gradient(self, p, egrad):
        return self.projection(p, egrad)
