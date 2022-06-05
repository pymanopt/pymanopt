import numpy as np
import scipy.linalg

from pymanopt.manifolds.manifold import Manifold, RetrAsExpMixin


class _PSDFixedRank(Manifold):
    def __init__(self, n, k, name, dimension):
        self._n = n
        self._k = k
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return 10 + self._k

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a.conj(),
            tangent_vector_b,
            axes=tangent_vector_a.ndim,
        ).real

    def dist(self, point_a, point_b):
        return self.norm(point_a, self.log(point_a, point_b))

    def norm(self, point, tangent_vector):
        return np.linalg.norm(tangent_vector)

    def projection(self, point, vector):
        YtY = point.T.conj() @ point
        AS = point.T.conj() @ vector - vector.T.conj() @ point
        Omega = scipy.linalg.solve_continuous_lyapunov(YtY, AS)
        return vector - point @ Omega

    to_tangent_space = projection

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return euclidean_gradient

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        return self.projection(point, euclidean_hessian)

    def exp(self, point, tangent_vector):
        return point + tangent_vector

    retraction = exp

    def log(self, point_a, point_b):
        u, _, vh = np.linalg.svd(point_b.T.conj() @ point_a)
        return point_b @ u @ vh - point_a

    def random_point(self):
        return np.random.normal(size=(self._n, self._k))

    def random_tangent_vector(self, point):
        random_vector = self.random_point()
        tangent_vector = self.projection(point, random_vector)
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def _normalize(self, array):
        return array / np.linalg.norm(array)

    def zero_vector(self, point):
        return np.zeros((self._n, self._k))


class PSDFixedRank(_PSDFixedRank):
    r"""Manifold of fixed-rank positive semidefinite (PSD) matrices.

    Args:
        n: Number of rows and columns of a point in the ambient space.
        k: Rank of matrices in the ambient space.

    Note:
        A point :math:`\vmX` on the manifold is parameterized as :math:`\vmX =
        \vmY\transp{\vmY}` where :math:`\vmY` is a real matrix of size ``n x
        k`` and rank ``k``.
        As such, :math:`\vmX` is symmetric, positive semidefinite with rank
        ``k``.

        Tangent vectors :math:`\dot{\vmY}` are represented as matrices of the
        same size as points on the manifold so that tangent vectors in the
        ambient space :math:`\R^{n \times n}` correspond to :math:`\dot{\vmX} =
        \vmY\transp{\dot{\vmY}} + \dot{\vmY}\transp{\vmY}`.
        The metric is the canonical Euclidean metric on :math:`\R^{n \times
        k}`.

        Since for any orthogonal matrix :math:`\vmQ` of size :math:`k \times k`
        it holds that :math:`\vmY\vmQ\transp{(\vmY\vmQ)} = \vmY\transp{\vmY}`,
        we identify all matrices of the form :math:`\vmY\vmQ` with an
        equivalence class.
        This set of equivalence classes then forms a Riemannian quotient
        manifold which is implemented here.

        Notice that this manifold is not complete: if optimization leads points
        to be rank-deficient, the geometry will break down.
        Hence, this geometry should only be used if it is expected that the
        points of interest will have rank exactly ``k``.
        Reduce ``k`` if that is not the case.

        The quotient geometry implemented here is the simplest case presented
        in [JBA+2010]_.
    """

    def __init__(self, n: int, k: int):
        name = f"Quotient manifold of {n}x{n} psd matrices of rank {k}"
        dimension = int(k * n - k * (k - 1) / 2)
        super().__init__(n, k, name, dimension)


class PSDFixedRankComplex(_PSDFixedRank):
    r"""Manifold of fixed-rank Hermitian positive semidefinite (PSD) matrices.

    Args:
        n: Number of rows and columns of a point in the ambient space.
        k: Rank of matrices in the ambient space.

    Note:
        A point :math:`\vmX` on the manifold is parameterized as :math:`\vmX =
        \vmY\conj{\vmY}`, where :math:`\vmY` is a complex matrix of size ``n x
        k`` and rank ``k``.

        Tangent vectors are represented as matrices of the same shape as points
        on the manifold.

        For any point :math:`\vmY` on the manifold, given any complex unitary
        matrix :math:`\vmU \in \C^{k \times k}`, we say
        :math:`\vmY\vmU` is equivalent to :math:`\vmY` since :math:`\vmY`
        and :math:`\vmY\vmU` are indistinguishable in the ambient space
        :math:`\C^{n \times n}`, i.e., :math:`\vmX = \vmY\vmU\conj{(\vmY\vmU)} =
        \vmY\conj{\vmY}`.
        Therefore, the set of equivalence classes forms a Riemannian
        quotient manifold :math:`\C^{n \times k} / \U(k)` where :math:`\U(k)`
        denotes the unitary group.
        The metric is the usual real trace inner product.

        Notice that this manifold is not complete: if optimization leads points
        to be rank-deficient, the geometry will break down.
        Hence, this geometry should only be used if it is expected that the
        points of interest will have rank exactly ``k``.
        Reduce ``k`` if that is not the case.

        The implementation follows the quotient geometry originally described
        in [Yat2013]_.
    """

    def __init__(self, n, k):
        name = f"Quotient manifold of Hermitian {n}x{n} matrices of rank {k}"
        dimension = 2 * k * n - k * k
        super().__init__(n, k, name, dimension)

    def random_point(self):
        return np.random.normal(
            size=(self._n, self._k)
        ) + 1j * np.random.normal(size=(self._n, self._k))


class Elliptope(Manifold, RetrAsExpMixin):
    r"""Manifold of fixed-rank PSD matrices with unit diagonal elements.

    Args:
        n: Number of rows and columns of a point in the ambient space.
        k: Rank of matrices in the ambient space.

    Note:
        A point :math:`\vmX` on the manifold is parameterized as :math:`\vmX =
        \vmY\transp{\vmY}` where :math:`\vmY` is a matrix of size ``n x k`` and
        rank ``k``.
        As such, :math:`\vmX` is symmetric, positive semidefinite with rank
        ``k``.

        Tangent vectors are represented as matrices of the same size as points
        on the manifold so that tangent vectors in the ambient space are of the
        form :math:`\dot{\vmX} = \vmY \transp{\dot{\vmY}} +
        \dot{\vmY}\transp{\vmY}` and :math:`\dot{X}_{ii} = 0`.
        The metric is the canonical Euclidean metric on :math:`\R^{n \times
        k}`.

        The diagonal constraints on :math:`X_{ii} = 1` translate to unit-norm
        constraints on the rows of :math:`\vmY`: :math:`\norm{\vmy_i} = 1`
        where :math:`\vmy_i` denotes the i-th column of :math:`\transp{\vmY}`.
        Without any further restrictions, this coincides with the oblique
        manifold (see :class:`pymanopt.manifolds.oblique.Oblique`).
        However, since for any orthogonal matrix :math:`\vmQ` of size ``k``, it
        holds that :math:`\vmY\vmQ\transp{(\vmY\vmQ)} = \vmY\transp{\vmY}`, we
        "group" all matrices of the form :math:`\vmY\vmQ` in an equivalence
        class.
        This set of equivalence classes is a Riemannian quotient manifold that
        is implemented here.

        Note that this geometry formally breaks down at rank-deficient points.
        This does not appear to be a major issue in practice when optimization
        algorithms converge to rank-deficient points, but convergence theorems
        no longer hold.
        As an alternative, you may try using the oblique manifold since it does
        not break down at rank drop.

        The geometry is taken from [JBA+2010]_.
    """

    def __init__(self, n, k):
        self._n = n
        self._k = k

        name = (
            f"Quotient manifold of {n}x{n} psd matrices of rank {k} "
            "with unit diagonal elements"
        )
        dimension = int(n * (k - 1) - k * (k - 1) / 2)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return 10 * self._k

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        return np.tensordot(
            tangent_vector_a, tangent_vector_b, axes=tangent_vector_a.ndim
        )

    def norm(self, point, tangent_vector):
        return np.sqrt(
            self.inner_product(point, tangent_vector, tangent_vector)
        )

    def projection(self, point, vector):
        eta = self._project_rows(point, vector)
        YtY = point.T @ point
        AS = point.T @ eta - vector.T @ point
        Omega = scipy.linalg.solve_continuous_lyapunov(YtY, -AS)
        return eta - point @ (Omega - Omega.T) / 2

    to_tangent_space = projection

    def retraction(self, point, tangent_vector):
        return self._normalize_rows(point + tangent_vector)

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return self._project_rows(point, euclidean_gradient)

    def euclidean_to_riemannian_hessian(
        self, point, euclidean_gradient, euclidean_hessian, tangent_vector
    ):
        scaling_grad = (euclidean_gradient * point).sum(axis=1)
        hess = euclidean_hessian - tangent_vector * scaling_grad[:, np.newaxis]
        scaling_hess = (
            tangent_vector * euclidean_gradient + point * euclidean_hessian
        ).sum(axis=1)
        hess -= point * scaling_hess[:, np.newaxis]
        return self.projection(point, hess)

    def random_point(self):
        return self._normalize_rows(np.random.normal(size=(self._n, self._k)))

    def random_tangent_vector(self, point):
        tangent_vector = self.projection(point, self.random_point())
        return tangent_vector / self.norm(point, tangent_vector)

    def transport(self, point_a, point_b, tangent_vector_a):
        return self.projection(point_b, tangent_vector_a)

    def _normalize_rows(self, array):
        return array / np.linalg.norm(array, axis=1)[:, np.newaxis]

    def _project_rows(self, point, vector):
        inner_products = (point * vector).sum(axis=1)
        return vector - point * inner_products[:, np.newaxis]

    def zero_vector(self, point):
        return np.zeros((self._n, self._k))
