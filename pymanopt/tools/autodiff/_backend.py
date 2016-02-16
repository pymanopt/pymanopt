class Backend(object):
    def __id(self, objective, argument):
        return objective

    compile_function = compute_gradient = compute_hessian = __id
