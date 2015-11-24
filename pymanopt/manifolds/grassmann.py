# Factory class for the Grassmann manifold. This is the manifold of p-
# dimensional subspaces of n dimensional real vector space. Initiation requires
# the dimensions n, p to be specified.
import numpy as np

from manifold import Manifold

class Grassmann(Manifold):

    def __init__(self, height, width):
        # Check that n is greater than or equal to p
        assert height >= width, ("Need n >= p. Values supplied were n = %d and "
                                "p = %d." % (height, width))

        # Set the dimensions of the Grassmann
        self.n = height
        self.p = width

    # Retract to the Grassmann using the qr decomposition of X + G
    def retr(self, X, G):
        # Calculate 'thin' qr decomposition of X + G
        q, r = np.linalg.qr(X + G)
        return q

    def egrad2rgrad(self, X, G):
        # Project G into the tangent space
        GNew = G - np.dot(X, np.dot(X.T, G))
        return GNew

    def norm(self, X, G):
        # Norm on the tangent space is simply the Euclidean norm.
        return np.linalg.norm(G)

    # Generate random Grassmann point using qr of random normally distributed
    # matrix.
    def rand(self):
        X = np.random.randn(self.n,self.p)
        q, r = np.linalg.qr(X)
        return q
