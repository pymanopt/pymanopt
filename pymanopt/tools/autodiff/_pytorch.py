"""
Module containing functions to differentiate functions using autograd.
"""
try:
    import torch
    import numpy as np
except ImportError:
    torch = None
    np = None

from ._backend import Backend, assert_backend_available


# The pytorch tape-based automatic differentiation means
# that one needs to compute the function to compute its gradient.
# Alas manopt uses different functions to compute the cost,
# its gradient, or its hessian. Therefore it is important to
# cache previous computations with the same x.


class PytorchBackend(Backend):
    def __str__(self):
        return "pytorch"

    def is_available(self):
        return torch is not None and np is not None

    @assert_backend_available
    def is_compatible(self, objective, arg):
        """
        To select the pytorch backend, use
        'Problem(manifold=..,cost=...,arg=torch.Tensor())'
        The tensor passed as argument is used as a python object
        to cache recent calls to the cost, egrad, or ehess functions
        """
        return callable(objective) and \
            isinstance(arg, torch.Tensor) and arg.nelement() == 0

    @assert_backend_available
    def _compile(self, objective, cache):
        assert isinstance(cache, torch.Tensor) and cache.nelement() == 0
        if hasattr(cache, 'cost'):
            return cache
        cache.x = None    # list woith torch copies of input np.arrays
        cache.ids = None  # list with ids of the input np.arrays
        cache.f = None    # scalar tensor with cost function
        cache.df = None   # list of gradient tensors

        def _astensor(x):
            return x.detach() if isinstance(x, torch.Tensor) \
                else torch.from_numpy(np.array(x))

        def _asiterable(x):
            return (x, True) if type(x) in (list, tuple) \
                else ([x], False)

        def _notcached(x, cache):
            if not cache.ids:
                return True
            if len(x) != len(cache.ids):
                return True
            for (xi, cachex, cacheid) in zip(x, cache.x, cache.ids):
                if (id(xi) != cacheid):
                    return True
                if not (_astensor(xi) == _astensor(cachex)).all().item():
                    return True
            return False

        def _updatex(x, cache):
            if _notcached(x, cache):
                cache.x = [_astensor(xi).clone().requires_grad_() for xi in x]
                cache.ids = [id(xi) for xi in x]
                cache.f = None
                cache.df = None

        def _updatef(seqp, cache):
            if not cache.f:
                cache.f = objective(cache.x) if seqp else objective(cache.x[0])
                if not torch.is_tensor(cache.f) or len(cache.f.size()) > 0:
                    raise ValueError("Pytorch backend wants a functions "
                                     "that returns a zerodim tensor(scalar)")

        def cost(x):
            xx, seqp = _asiterable(x)
            _updatex(xx, cache)
            _updatef(seqp, cache)
            return cache.f.item()

        def _updatedf(seqp, cache):
            if not cache.df:
                _updatef(seqp, cache)
                cache.df = torch.autograd.grad(cache.f, cache.x,
                                               create_graph=True,
                                               allow_unused=True)

        def egrad(x):
            xx, seqp = _asiterable(x)
            _updatex(xx, cache)
            _updatedf(seqp, cache)
            return [di.detach().numpy() for di in cache.df] if seqp \
                else cache.df[0].detach().numpy()

        def ehess(x, u):
            xx, seqp = _asiterable(x)
            uu, sequ = _asiterable(u)
            if seqp != sequ or len(xx) != len(uu):
                raise ValueError("Incompatible lists in ehess")
            _updatex(xx, cache)
            _updatedf(seqp, cache)
            r = 0
            for (di, ui) in zip(cache.df, uu):
                n = di.nelement()
                r = r + torch.dot(di.view(n), _astensor(ui).view(n))
            h = torch.autograd.grad([r], cache.x,
                                    retain_graph=True, allow_unused=True)
            return [hi.numpy() for hi in h] if seqp else h[0].numpy()

        cache.cost = cost
        cache.egrad = egrad
        cache.ehess = ehess
        return cache

    @assert_backend_available
    def compile_function(self, objective, argument):
        return self._compile(objective, argument).cost

    @assert_backend_available
    def compute_gradient(self, objective, argument):
        return self._compile(objective, argument).egrad

    @assert_backend_available
    def compute_hessian(self, objective, argument):
        return self._compile(objective, argument).ehess
