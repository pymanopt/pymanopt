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
        raise NotImplementedError()

    def rand(self):
        U = self._stiefel_m.rand()
        S = np.diag(np.sort(np.random.rand(5))[::-1])
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

        return (Up, Z[1], Vp)

    def norm(self, X, G):
        return np.sqrt(self.inner(X, G, G))

    def randvec(self, X):
        Up = np.random.randn(self._m, self._k)
        Vp = np.random.randn(self._n, self._k)
        M = np.random.randn(self._k, self._k)

        Z = self._tangent(X, (Up, M, Vp))

        nrm = self.norm(X, Z)

        return (Z[0]/nrm, Z[1]/nrm, Z[2]/nrm)

    def inner(self, X, G, H):
        # Einsum used in this way is equivalent to tensordot but slightly
        # faster.

        return np.sum(np.tensordot(a, b) for (a, b) in zip(G, H))
