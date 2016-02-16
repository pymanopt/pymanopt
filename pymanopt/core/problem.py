"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""
import theano.tensor as T

from pymanopt.tools.autodiff import TheanoBackend, AutogradBackend


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
    def __init__(self, man=None, cost=None, grad=None, hess=None, egrad=None,
                 ehess=None, arg=None, precon=None, verbosity=2):
        self.man = man
        self.cost = cost
        self.grad = grad
        self.hess = hess
        self.egrad = egrad
        self.ehess = ehess
        self.arg = arg

        if precon is None:
            def precon(x, d):
                return d
        self.precon = precon

        self.verbosity = verbosity

    def prepare(self, need_grad=False, need_hess=False):
        """
        Function to prepare the problem for solving, this will be
        executed by the solver before optimization to compile a
        cost and/or compute the grad and hess of the cost as required.

        The arguments need_grad and need_hess are used to specify
        whether grad and hess are required by the solver.
        """
        self._finalize(need_grad, need_hess)

        if need_grad and self.grad is None:
            self.grad = lambda x: self.man.egrad2rgrad(x, self.egrad(x))
            # Assume if Hessian is needed gradient is as well
            if need_hess and self.hess is None:
                self.hess = lambda x, a: self.man.ehess2rhess(
                    x, self.egrad(x), self.ehess(x, a), a)

    def _finalize(self, need_grad, need_hess):
        # Conditionally load autodiff backend if needed.
        if isinstance(self.cost, T.TensorVariable):
            if not isinstance(self.arg, T.TensorVariable):
                raise ValueError(
                    "Theano backend requires an argument with respect to "
                    "which compilation of the cost function is to be carried "
                    "out")
            backend = TheanoBackend()
        elif callable(self.cost):
            backend = AutogradBackend()
        else:
            raise ValueError("Cannot identify autodiff backend from cost "
                             "variable.")

        if self.verbosity >= 1:
            print("Compiling cost function...")
        compiled_cost_function = backend.compile_function(self.cost, self.arg)

        if need_grad and self.egrad is None and self.grad is None:
            if self.verbosity >= 1:
                print("Computing gradient of cost function...")
            self.egrad = backend.compute_gradient(self.cost, self.arg)

        if need_hess and self.ehess is None and self.hess is None:
            if self.verbosity >= 1:
                print("Computing Hessian of cost function...")
            self.ehess = backend.compute_hessian(self.cost, self.arg)

        self.cost = compiled_cost_function
