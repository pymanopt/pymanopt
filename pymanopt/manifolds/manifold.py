# Abstract base class setting out template for manifold classes

import abc

class Manifold(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def retr(self, X, G):
        # A retraction mapping from the tangent space at X to the manifold.
        # See Absil for definition of retraction.
        raise NotImplementedError()

    @abc.abstractmethod
    def egrad2rgrad(self, X, G):
        # A mapping from the Euclidean gradient G into the tangent space
        # to the manifold at X.
        raise NotImplementedError()
    
    @abc.abstractmethod
    def norm(self, X, G):
        # Compute the norm of a tangent vector G, which is tangent to the
        # manifold at X.
        raise NotImplementedError()
        
    @abc.abstractmethod
    def rand(self):
        # A function which returns a random point on the manifold.
        raise NotImplementedError()