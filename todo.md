# pymanopt todo list

- manifolds

    - implement tests for manifolds

    - implement further manifolds

    - Stiefel
        - [todo @118 (simplify expressions)](./pymanopt/manifolds/stiefel.py#L118) Not sure if this is really possible though...

- solvers

    - nelder_mead
        [todo @32](./pymanopt/solvers/nelder_mead.py#L32) need to decide what to do about the TR iterations

    - implement tests for solvers

- tools

    - autodiff/_tensorflow
        - Implement compute_hessian [todo @62](./pymanopt/tools/autodiff/_tensorflow.py#L62)
        - Add tests

    - autodiff/_theano
        - Fix theano's no Rop fallback for the product manifold/investigate whether this no Rop fallback is really necessary

    - autodiff/_autograd
        - Need to change to hessian_vector_product, but not working yet, due to an issue with autograd. Issue has been fixed but they haven't merged the fix into master branch yet, apparently will happen very soon (see [here](https://github.com/HIPS/autograd/issues/86)). Have implemented our own fix for now, should be removed and replaced with theirs at some point in future.
