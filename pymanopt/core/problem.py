"""
Module containing pymanopt problem class. Use this to build a problem
object to feed to one of the solvers.
"""
from pymanopt.tools import theano_functions as tf


class Problem:
    """
    Problem class for setting up a problem to feed to one of the
    pymanopt solvers.

    Attributes:
        - man
            Manifold to optimize over.
        - cost
            Function which takes an element of man and returns a real
            number. The solver will attempt to minimise this function.
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
            The 'Euclidean hessian', ehess(x, a) should return the
            directional derivative of egrad at x in direction a. This
            need not lie in the tangent space.
        - ad_cost/ad_arg
            These allow you to define a cost in theano (also planned are
            autograd and tensorflow) whose gradient (and hessian if
            necessary) will automatically be computed.
            We recommend you take this approach rather than calculating
            gradients and hessians by hand.
            ad_cost is the (scalar) cost and ad_arg is the (tensor)
            variable with respect to which you would like to
            optimize. Their type define the autodiff backend used.
    """
    def __init__(self, man=None, cost=None, grad=None,
                 hess=None, egrad=None, ehess=None,
                 ad_cost=None, ad_arg=None):
        self.man = man
        self.cost = cost
        self.grad = grad
        self.hess = hess
        self.egrad = egrad
        self.ehess = ehess
        self.ad_cost = ad_cost
        self.ad_arg = ad_arg

    def prepare(self, need_grad=False, need_hess=False):
        """
        Function to prepare the problem for solving, this will be
        executed by the solver before optimization to compile a
        cost and/or compute the grad and hess of the cost as required.

        The arguments need_grad and need_hess are used to specify
        whether grad and hess are required by the solver.
        """
        if self.cost is None:
            self.cost = tf.compile(self.ad_cost, self.ad_arg)

        if need_grad and self.grad is None:
            if self.egrad is None:
                self.egrad = tf.gradient(self.ad_cost, self.ad_arg)
            self.grad = lambda x: self.man.egrad2rgrad(x, self.egrad(x))

        if need_hess and self.hess is None:
            if self.ehess is None:
                self.ehess = tf.hess(self.ad_cost, self.ad_arg)
            self.hess = lambda x, a: self.man.ehess2rhess(
                    x, self.egrad(x), self.ehess(x, a), a)
