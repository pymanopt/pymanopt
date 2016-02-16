import theano.tensor as T

from pymanopt.tools.autodiff import _theano, _autograd


def compile(problem, need_grad, need_hess):
    # Conditionally load autodiff backend if needed.
    if isinstance(problem.cost, T.TensorVariable):
        if not isinstance(problem.arg, T.TensorVariable):
            raise ValueError(
                "Theano backend requires an argument with respect to "
                "which compilation of the cost function is to be carried "
                "out")
        backend = _theano
    elif callable(problem.cost):
        backend = _autograd
    else:
        raise ValueError("Cannot identify autodiff backend from cost "
                         "variable.")

    if problem.verbosity >= 1:
        print("Compiling cost function...")
    compiled_cost_function = backend.compile(problem.cost, problem.arg)

    if need_grad and problem.egrad is None and problem.grad is None:
        if problem.verbosity >= 1:
            print("Computing gradient of cost function...")
        problem.egrad = backend.gradient(problem.cost, problem.arg)

    if need_hess and problem.ehess is None and problem.hess is None:
        if problem.verbosity >= 1:
            print("Computing Hessian of cost function...")
        problem.ehess = backend.hessian(problem.cost, problem.arg)

    problem.cost = compiled_cost_function
