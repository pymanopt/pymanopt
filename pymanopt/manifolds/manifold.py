# Abstract base class setting out template for manifold classes

import abc

import numpy as np

class Manifold(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        # Name of the manifold.
        raise NotImplementedError

    @abc.abstractproperty
    def dim(self):
        # Dimension of the manifold
        raise NotImplementedError

    @abc.abstractproperty
    def typicaldist(self):
        # Returns the "scale" of the manifold. This is used by the trust-regions
        # solver, to determine default initial and maximal trust-region radii.
        raise NotImplementedError

    @abc.abstractmethod
    def dist(self, X, Y):
        # Geodesic distance on the manifold
        raise NotImplementedError

    @abc.abstractmethod
    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        raise NotImplementedError

    @abc.abstractmethod
    def proj(self, X, G):
        # Project into the tangent space. Usually the same as egrad2rgrad
        raise NotImplementedError

    @abc.abstractmethod
    def ehess2rhess(self, X, Hess):
        # Convert Euclidean hessian into Riemannian hessian.
        raise NotImplementedError

    @abc.abstractmethod
    def retr(self, X, G):
        # A retraction mapping from the tangent space at X to the manifold.
        # See Absil for definition of retraction.
        raise NotImplementedError

    @abc.abstractmethod
    def egrad2rgrad(self, X, G):
        # A mapping from the Euclidean gradient G into the tangent space
        # to the manifold at X.
        raise NotImplementedError

    @abc.abstractmethod
    def norm(self, X, G):
        # Compute the norm of a tangent vector G, which is tangent to the
        # manifold at X.
        raise NotImplementedError

    @abc.abstractmethod
    def rand(self):
        # A function which returns a random point on the manifold.
        raise NotImplementedError

    @abc.abstractmethod
    def randvec(self, X):
        # Returns a random, unit norm vector in the tangent space at X.
        raise NotImplementedError

    def lincomb(self, X, a1, u1, a2=None, u2=None):
        """
        Given a point X, two tangent vectors u1 and u2 at X, and two real
        coefficients a1 and a2, returns a tangent vector at X representing
        a1 * u1 + a2 * u2, if u1 and u2 are represented as matrices.

        If a2 and u2 are omitted, the returned tangent vector is a1 * u1.

        The input X is unused.
        """
        y = a1 * u1
        if a2 is not None and u2 is not None:
            return y + a2 * u2
        return y

    def zerovec(self, X):
        """
        Returns the zero tangent vector at X.
        """
        return np.zeros(np.shape(X))
