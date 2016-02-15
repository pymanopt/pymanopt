# Abstract base class setting out template for manifold classes

import abc

import numpy as np


class Manifold(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def name(self):
        # Name of the manifold
        pass

    @abc.abstractproperty
    def dim(self):
        # Dimension of the manifold
        pass

    @abc.abstractproperty
    def typicaldist(self):
        # Returns the "scale" of the manifold. This is used by the
        # trust-regions solver, to determine default initial and maximal
        # trust-region radii.
        pass

    @abc.abstractmethod
    def dist(self, X, Y):
        # Geodesic distance on the manifold
        pass

    @abc.abstractmethod
    def inner(self, X, G, H):
        # Inner product (Riemannian metric) on the tangent space
        pass

    @abc.abstractmethod
    def proj(self, X, G):
        # Project into the tangent space. Usually the same as egrad2rgrad
        pass

    @abc.abstractmethod
    def ehess2rhess(self, X, Hess):
        # Convert Euclidean hessian into Riemannian hessian.
        pass

    @abc.abstractmethod
    def retr(self, X, G):
        # A retraction mapping from the tangent space at X to the manifold.
        # See Absil for definition of retraction.
        pass

    @abc.abstractmethod
    def egrad2rgrad(self, X, G):
        # A mapping from the Euclidean gradient G into the tangent space
        # to the manifold at X.
        pass

    @abc.abstractmethod
    def norm(self, X, G):
        # Compute the norm of a tangent vector G, which is tangent to the
        # manifold at X.
        pass

    @abc.abstractmethod
    def rand(self):
        # A function which returns a random point on the manifold.
        pass

    @abc.abstractmethod
    def randvec(self, X):
        # Returns a random, unit norm vector in the tangent space at X.
        pass

    @abc.abstractmethod
    def transp(self, x1, x2, d):
        # Transports d, which is a tangent vector at x1, into the tangent
        # space at x2.
        pass

    @abc.abstractmethod
    def exp(self, X, U):
        # The exponential (in the sense of Lie group theory) of a tangent
        # vector U at X.
        pass

    @abc.abstractmethod
    def log(self, X, Y):
        # The logarithm (in the sense of Lie group theory) of Y. This is the
        # inverse of exp.
        pass

    def zerovec(self, X):
        """
        Returns the zero tangent vector at X.
        """
        return np.zeros(np.shape(X))
