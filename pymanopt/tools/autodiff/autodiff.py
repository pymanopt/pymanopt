from warnings import warn


def compile(problem, need_grad, need_hess):
    # Conditionally load autodiff backend if needed
    if (problem.cost is None or
       (need_grad and problem.grad is None and problem.egrad is None) or
       (need_hess and problem.hess is None and problem.ehess is None)):
        if type(problem.ad_cost).__name__ == 'TensorVariable':
            from pymanopt.tools.autodiff import _theano as ad
        elif type(problem.ad_cost).__name__ == 'function':
            from pymanopt.tools.autodiff import _autograd as ad
        else:
            warn('Cannot identify autodiff backend from '
                 'ad_cost variable type.')

    if problem.verbosity >= 1:
        print("Compiling cost function...")
    if problem.cost is None:
        problem.cost = ad.compile(problem.ad_cost, problem.ad_arg)

    if need_grad and problem.egrad is None and problem.grad is None:
        if problem.verbosity >= 1:
            print("Computing gradient of cost function...")
        problem.egrad = ad.gradient(problem.ad_cost, problem.ad_arg)
        # Assume if Hessian is needed gradient is as well
        if need_hess and problem.ehess is None and problem.hess is None:
            if problem.verbosity >= 1:
                print("Computing Hessian of cost function...")
            problem.ehess = ad.hessian(problem.ad_cost, problem.ad_arg)
