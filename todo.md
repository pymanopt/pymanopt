# pymanopt todo list

- tools

    - autodiff/_autograd
        - Need to change to hessian_vector_product, but not working yet, due to an issue with autograd. Issue has been fixed but they haven't merged the fix into master branch yet, apparently will happen very soon (see [here](https://github.com/HIPS/autograd/issues/86)). Have implemented our own fix for now, should be removed and replaced with theirs at some point in future.

        - [todo @20](./pymanopt/tools/autodiff/_autograd.py#L20)

- manifolds

    - stiefel
        - [todo @118 (simplify expressions)](./pymanopt/manifolds/stiefel.py#L118) Not sure if this is really possible though...

    - Implement tests for manifolds

- solvers

    - nelder_mead
        [todo @35](./pymanopt/solvers/nelder_mead.py#L35), [57](./pymanopt/solvers/nelder_mead.py#L57) need to decide what to do about the TR iterations

    - trust_regions
        - [todo @95 (implement value checks)](./pymanopt/solvers/trust_regions.py#L95)

    - Implement tests for solvers

- autodiff
    - add tensorflow support
