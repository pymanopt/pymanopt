"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""
from __future__ import print_function

import theano.tensor as T

from pymanopt.tools.autodiff import AutogradBackend, TheanoBackend


class Problem(object):
    """
    Problem class for setting up a problem to feed to one of the
    pymanopt solvers.

    Attributes:
        - man
            Manifold to optimize over.
        - cost
            A callable which takes an element of man and returns a real number,
            or a symbolic Theano or TensorFlow expression. In case of a
            symbolic expression, the gradient (and if necessary the Hessian)
            are computed automatically if they are not explicitly given. We
            recommend you take this approach rather than calculating gradients
            and Hessians by hand.
        - grad
            grad(x) is the gradient of cost at x. This must take an
            element X of man and return an element of the tangent space
            to man at X. This is usually computed automatically and
            doesn't need to be set by the user.
        - hess
            hess(x, a) is the directional derivative of grad at x, in
            direction a. It should return an element of the tangent
            space to man at x.
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
    _backend = None

    def __init__(self, man, cost, egrad=None, ehess=None, grad=None, hess=None,
                 arg=None, precon=None, verbosity=2):
        self.man = man
        # We keep a reference to the original cost function in case we want to
        # call the `prepare` method twice (for instance, after switching from
        # a first- to second-order method).
        self._cost = None
        self._original_cost = cost
        self._egrad = egrad
        self._ehess = ehess
        self._grad = grad
        self._hess = hess
        self._arg = arg

        if precon is None:
            def precon(x, d):
                return d
        self.precon = precon

        self.verbosity = verbosity

    def _get_backend(self):
        if self._backend is None:
            if isinstance(self._original_cost, T.TensorVariable):
                if not isinstance(self._arg, T.TensorVariable):
                    raise ValueError(
                        "Theano backend requires an argument with respect to "
                        "which compilation is to be carried out")
                if not callable(TheanoBackend):
                    raise ImportError(TheanoBackend)
                backend = TheanoBackend()
            elif callable(self._original_cost):
                if not callable(AutogradBackend):
                    raise ImportError(AutogradBackend)
                backend = AutogradBackend()
            else:
                raise ValueError(
                    "Cannot identify compilation backend from cost function "
                    "`{:s}`".format(self._original_cost))
            self._backend = backend
        return self._backend

    @property
    def cost(self):
        if self._cost is None:
            if self.verbosity >= 1:
                print("Compiling cost function...")
            backend = self._get_backend()
            self._cost = backend.compile_function(self._original_cost,
                                                  self._arg)
        return self._cost

    @property
    def egrad(self):
        if self._egrad is None:
            if self.verbosity >= 1:
                print("Computing gradient of cost function...")
            backend = self._get_backend()
            self._egrad = backend.compute_gradient(self._original_cost,
                                                   self._arg)
        return self._egrad

    @property
    def grad(self):
        if self._grad is None:
            # Explicit access forces computation/compilation if necessary.
            egrad = self.egrad

            def grad(x):
                return self.man.egrad2rgrad(x, egrad(x))
            self._grad = grad
        return self._grad

    @property
    def ehess(self):
        if self._ehess is None:
            if self.verbosity >= 1:
                print("Computing Hessian of cost function...")
            backend = self._get_backend()
            self._ehess = backend.compute_hessian(self._original_cost,
                                                  self._arg)
        return self._ehess

    @property
    def hess(self):
        if self._hess is None:
            # Explicit access forces computation if necessary.
            ehess = self.ehess

            def hess(x, a):
                return self.man.ehess2rhess(
                    x, self.egrad(x), self.ehess(x, a), a)
            self._hess = hess
        return self._hess
