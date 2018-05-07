"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""
from __future__ import print_function

from pymanopt.tools.autodiff import Function


class Problem(object):
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
        - arg
            A symbolic (tensor) variable with respect to which you would like
            to optimize. Its type (together with the type of the cost argument)
            defines the autodiff backend used.
        - verbosity (2)
            Level of information printed by the solver while it operates, 0
            is silent, 2 is most information.
    """
    def __init__(self, manifold, cost, egrad=None, ehess=None, grad=None,
                 hess=None, arg=None, precon=None, verbosity=2):
        self._cost = None
        self._egrad = None
        self._ehess = None
        self._grad = None
        self._hess = None

        self.manifold = manifold
        self._original_cost = cost
        self._original_egrad = egrad
        self._original_ehess = ehess
        self._original_grad = grad
        self._original_hess = hess
        self._arg = arg

        if precon is None:
            def precon(x, d):
                return d
        self.precon = precon

        self.verbosity = verbosity

    @property
    def cost(self):
        if self._cost is None:
            self._cost = Function(self._original_cost, self._arg)
        return self._cost

    # FIXME: Since _arg is passed to the problem class, we can't have different
    #        types of functions/call graphs for cost, gradients and Hessians.

    @property
    def egrad(self):
        if self._egrad is None:
            if self._original_egrad is None:
                self._egrad = self.cost.compute_gradient()
            else:
                self._egrad = Function(self._original_egrad, self._arg)
        return self._egrad

    @property
    def grad(self):
        if self._grad is None:
            if self._original_grad is None:
                egrad = self.egrad

                def grad(x):
                    return self.manifold.egrad2rgrad(x, egrad(x))
                self._grad = Function(grad)
            else:
                self._grad = Function(self._original_grad, self._arg)
        return self._grad

    @property
    def ehess(self):
        if self._ehess is None:
            if self._original_ehess is None:
                self._ehess = self.cost.compute_hessian()
            else:
                self._ehess = Function(self._original_ehess, self._arg)
        return self._ehess

    @property
    def hess(self):
        if self._hess is None:
            if self._original_hess is None:
                ehess = self.ehess

                def hess(x, a):
                    return self.manifold.ehess2rhess(
                        x, self.egrad(x), ehess(x, a), a)
                self._hess = Function(hess)
            else:
                self._hess = Function(self._original_hess, self._arg)
        return self._hess
