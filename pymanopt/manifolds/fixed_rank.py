"""
Module containing manifolds of fixed rank matrices.
"""

import numpy as np

from pymanopt.manifolds.manifold import Manifold
from pymanopt.manifolds import Stiefel


class FixedRankEmbedded(Manifold):
    """
    Manifold struct to optimize fixed-rank matrices w/ an embedded geometry.

    FixedRankEmbedded(m, n, k)

    Manifold of m-by-n real matrices of fixed rank k. This follows the
    embedded geometry described in Bart Vandereycken's 2013 paper:
    "Low-rank matrix completion by Riemannian optimization".

    Paper link: http://arxiv.org/pdf/1209.3834.pdf

    A point X on the manifold is represented as a structure with three
    fields: U, S and V. The matrices U (mxk) and V (nxk) are orthonormal,
    while the matrix S (kxk) is any diagonal, full rank matrix.
    Following the SVD formalism, X = U*S*V'. Note that the diagonal entries
    of S are not constrained to be nonnegative.

    Tangent vectors are represented as a structure with three fields: Up, M
    and Vp. The matrices Up (mxk) and Vp (mxk) obey Up'*U = 0 and Vp'*V = 0.
    The matrix M (kxk) is arbitrary. Such a structure corresponds to the
    following tangent vector in the ambient space of mxn matrices:
      Z = U*M*V' + Up*V' + U*Vp'
    where (U, S, V) is the current point and (Up, M, Vp) is the tangent
    vector at that point.

    Vectors in the ambient space are best represented as mxn matrices. If
    these are low-rank, they may also be represented as structures with
    U, S, V fields, such that Z = U*S*V'. Their are no resitrictions on what
    U, S and V are, as long as their product as indicated yields a real, mxn
    matrix.

    The chosen geometry yields a Riemannian submanifold of the embedding
    space R^(mxn) equipped with the usual trace (Frobenius) inner product.


    Please cite the Manopt paper as well as the research paper:
        @Article{vandereycken2013lowrank,
          Title   = {Low-rank matrix completion by {Riemannian} optimization},
          Author  = {Vandereycken, B.},
          Journal = {SIAM Journal on Optimization},
          Year    = {2013},
          Number  = {2},
          Pages   = {1214--1236},
          Volume  = {23},
          Doi     = {10.1137/110845768}
        }

    See also: fixedrankfactory_2factors fixedrankfactory_3factors

    This file is based on fixedrankembeddedfactory from Manopt: www.manopt.org.
    Ported by: Jamie Townsend, Sebastian Weichwald
    Original author: Nicolas Boumal, Dec. 30, 2012.
    Contributors:
    Change log:

      Feb. 20, 2014 (NB):
          Added function tangent to work with checkgradient.

      June 24, 2014 (NB):
          A couple modifications following
          Bart Vandereycken's feedback:
          - The checksum (hash) was replaced for a faster alternative: it's a
            bit less "safe" in that collisions could arise with higher
            probability, but they're still very unlikely.
          - The vector transport was changed.
          The typical distance was also modified, hopefully giving the
          trustregions method a better initial guess for the trust region
          radius, but that should be tested for different cost functions too.

       July 11, 2014 (NB):
          Added ehess2rhess and tangent2ambient, supplied by Bart.

       July 14, 2014 (NB):
          Added vec, mat and vecmatareisometries so that hessianspectrum now
          works with this geometry. Implemented the tangent function.
          Made it clearer in the code and in the documentation in what format
          ambient vectors may be supplied, and generalized some functions so
          that they should now work with both accepted formats.
          It is now clearly stated that for a point X represented as a
          triplet (U, S, V), the matrix S needs to be diagonal.
    """

    def __init__(self, m, n, k):
        self._m = m
        self._n = n
        self._k = k

        self._name = ("Manifold of {m}-by-{n} matrices with rank {k} and "
                      "embedded geometry".format(m=m, n=n, k=k))

        self._stiefel_m = Stiefel(m, k)
        self._stiefel_n = Stiefel(n, k)

    def __str__(self):
        return self._name

    @property
    def dim(self):
        return (self._m + self._n - self._k) * self._k

    @property
    def typicaldist(self):
        return self.dim

    def dist(self, X, Y):
        raise NotImplementedError

    def inner(self, X, G, H):
        # Einsum used in this way is equivalent to tensordot but slightly
        # faster.

        return np.sum(np.tensordot(a, b) for (a, b) in zip(G, H))

    def _apply_ambient(self, Z, W):
        """
        For a given ambient vector Z, given as a tuple (U, S, V) such that
        Z = U*S*V', applies it to a matrix W to calculate the matrix product
        ZW.
        """
        if isinstance(Z, tuple):
            return np.dot(Z[0], np.dot(Z[1], np.dot(Z[2].T, W)))
        else:
            return np.dot(Z, W)

    def _apply_ambient_transpose(self, Z, W):
        """
        Same as apply_ambient, but applies Z' to W.
        """
        if isinstance(Z, tuple):
            return np.dot(Z[2], np.dot(Z[1], np.dot(Z[0].T, W)))
        else:
            return np.dot(Z.T, W)

    def proj(self, X, Z):
        """
        Note that Z must either be an m x n matrix from the ambient space, or
        else a tuple (Uz, Sz, Vz), where Uz * Sz * Vz is in the ambient space
        (of low-rank matrices).

        This function then returns a tangent vector parameterized as
        (Up, M, Vp), as described in the class docstring.
        """
        ZV = self._apply_ambient(Z, X[2])
        UtZV = np.dot(X[0].T, ZV)
        ZtU = self._apply_ambient_transpose(Z, X[0])

        Up = ZV - np.dot(X[0], UtZV)
        M = UtZV
        Vp = ZtU - np.dot(X[2], UtZV.T)

        return _TangentVector((Up, M, Vp))

    egrad2rgrad = proj

    def ehess2rhess(self, X, egrad, ehess, H):
        # Euclidean part
        Up, M, Vp = self.proj(X, ehess)

        # Curvature part
        T = self._apply_ambient(egrad, H[2]) / np.diag(X[1])
        Up += T - np.dot(X[0], np.dot(X[0].T, T))
        T = self._apply_ambient_transpose(egrad, H[0]) / np.diag(X[1])
        Vp += T - np.dot(X[2], np.dot(X[2].T, T))

        return _TangentVector((Up, M, Vp))

    # This retraction is second order, following general results from
    # Absil, Malick, "Projection-like retractions on matrix manifolds",
    # SIAM J. Optim., 22 (2012), pp. 135-158.
    def retr(self, X, Z):
        Qu, Ru = np.linalg.qr(Z[0])
        Qv, Rv = np.linalg.qr(Z[2])

        T = np.bmat([[X[1] + Z[1], Rv.T],
                     [Ru, np.zeros((self._k, self._k))]])

        # Numpy svd outputs St as a 1d vector, not a matrix.
        (Ut, St, Vt) = np.linalg.svd(T, full_matrices=False)

        U = np.dot(np.bmat([X[0], Qu]), Ut[:, :self._k])
        V = np.dot(np.bmat([X[2], Qv]), Vt[:, :self._k])
        S = np.diag(St[:self._k]) + np.spacing(1) * np.eye(self._k)
        return (U, S, V)

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def rand(self):
        U = self._stiefel_m.rand()
        S = np.diag(np.sort(np.random.rand(self._k))[::-1])
        V = self._stiefel_n.rand()
        return (U, S, V)

    def _tangent(self, X, Z):
        """
        Given Z in tangent vector format, projects the components Up and Vp
        such that they satisfy the tangent space constraints up to numerical
        errors. If Z was indeed a tangent vector at X, this should barely
        affect Z (it would not at all if we had infinite numerical accuracy).
        """
        Up = Z[0] - np.dot(X[0], np.dot(X[0].T, Z[0]))
        Vp = Z[2] - np.dot(X[2], np.dot(X[2].T, Z[2]))

        return _TangentVector((Up, Z[1], Vp))

    def randvec(self, X):
        Up = np.random.randn(self._m, self._k)
        Vp = np.random.randn(self._n, self._k)
        M = np.random.randn(self._k, self._k)

        Z = self._tangent(X, (Up, M, Vp))

        nrm = self.norm(X, Z)

        return _TangentVector((Z[0]/nrm, Z[1]/nrm, Z[2]/nrm))

    def tangent2ambient(self, X, Z):
        """
        Transforms a tangent vector Z represented as a structure (Up, M, Vp)
        into a structure with fields (U, S, V) that represents that same
        tangent vector in the ambient space of mxn matrices, as U*S*V'.
        This matrix is equal to X.U*Z.M*X.V' + Z.Up*X.V' + X.U*Z.Vp'. The
        latter is an mxn matrix, which could be too large to build
        explicitly, and this is why we return a low-rank representation
        instead. Note that there are no guarantees on U, S and V other than
        that USV' is the desired matrix. In particular, U and V are not (in
        general) orthonormal and S is not (in general) diagonal.
        (In this implementation, S is identity, but this might change.)
        """
        U = np.bmat([np.dot(X[0], Z[1]) + Z[0], X[0]])
        S = np.eye(2 * self._k)
        V = np.bmat([X[2], Z[2]])
        return (U, S, V)

    # Comment from Manopt:
    # New vector transport on June 24, 2014 (as indicated by Bart)
    # Reference: Absil, Mahony, Sepulchre 2008 section 8.1.3:
    # For Riemannian submanifolds of a Euclidean space, it is acceptable to
    # transport simply by orthogonal projection of the tangent vector
    # translated in the ambient space.
    def transp(self, X1, X2, G):
        return self.proj(X2, self.tangent2ambient(X1, G))

    def exp(self, X, U):
        raise NotImplementedError

    def log(self, X, Y):
        raise NotImplementedError

    def pairmean(self, X, Y):
        raise NotImplementedError


class _TangentVector(tuple):
    def __add__(self, other):
        return _TangentVector((s + o for (s, o) in zip(self, other)))

    def __sub__(self, other):
        return _TangentVector((s - o for (s, o) in zip(self, other)))

    def __mul__(self, other):
        return _TangentVector((other * s for s in self))

    __rmul__ = __mul__

    def __div__(self, other):
        return _TangentVector((val / other for val in self))

    def __neg__(self):
        return _TangentVector((-val for val in self))
