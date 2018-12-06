"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""

from pymanopt.tools._functions import CallableFunction


class Problem:
    """
    Problem class for setting up a problem to feed to one of the
    pymanopt solvers.

    Attributes:
        - manifold
            Manifold to optimize over.
        - cost
            A callable which takes an element of manifold and returns a
            real number, or a symbolic Theano or TensorFlow expression.
            In case of a symbolic expression, the gradient (and if
            necessary the Hessian) are computed automatically if they are
            not explicitly given. We recommend you take this approach
            rather than calculating gradients and Hessians by hand.
        - grad
            grad(x) is the gradient of cost at x. This must take an
            element X of manifold and return an element of the tangent space
            to manifold at X. This is usually computed automatically and
            doesn't need to be set by the user.
        - hess
            hess(x, a) is the directional derivative of grad at x, in
            direction a. It should return an element of the tangent
            space to manifold at x.
        - egrad
            The 'Euclidean gradient', egrad(x) should return the grad of
            cost in the usual sense, i.e. egrad(x) need not lie in the
            tangent space.
        - ehess
            The 'Euclidean Hessian', ehess(x, a) should return the
            directional derivative of egrad at x in direction a. This
            need not lie in the tangent space.
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    """
    def __init__(self, manifold, cost, egrad=None, ehess=None, grad=None,
                 hess=None, precon=None, verbosity=2):
        self.manifold = manifold
        self.cost = cost

        self._egrad = egrad
        self._ehess = ehess
        self._grad = grad
        self._hess = hess

        if precon is None:
            def precon(x, d):
                return d
        self.precon = precon

        self.verbosity = verbosity

    @property
    def egrad(self):
        if self._egrad is None:
            self._egrad = self.cost.compute_gradient()
        return self._egrad

    @property
    def grad(self):
        if self._grad is None:
            egrad = self.egrad

            def grad(x):
                return self.manifold.egrad2rgrad(x, egrad(x))
            self._grad = CallableFunction(grad)
        return self._grad

    @property
    def ehess(self):
        if self._ehess is None:
            self._ehess = self.cost.compute_hessian()
        return self._ehess

    @property
    def hess(self):
        if self._hess is None:
            ehess = self.ehess

            def hess(x, a):
                return self.manifold.ehess2rhess(
                    x, self.egrad(x), ehess(x, a), a)
            self._hess = CallableFunction(hess)
        return self._hess
