# pymanopt todo list

- manifolds

    - implement tests for manifolds

    - implement further manifolds

    - Stiefel
        - [todo @121 (simplify expressions)](./pymanopt/manifolds/stiefel.py#L121) Not sure if this is really possible though...

- solvers

    - nelder_mead
        [todo @37](./pymanopt/solvers/nelder_mead.py#L37) need to decide what to do about the TR iterations

    - implement tests for solvers

- tools

    - add tensorflow support

    - autodiff/_autograd
        - Need to change to hessian_vector_product, but not working yet, due to an issue with autograd. Issue has been fixed but they haven't merged the fix into master branch yet, apparently will happen very soon (see [here](https://github.com/HIPS/autograd/issues/86)). Have implemented our own fix for now, should be removed and replaced with theirs at some point in future.