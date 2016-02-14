# pymanopt todo list

- tools

    - autodiff/_autograd
        - [todo @26 (check whether hessian_vector_product should be used instead)](./pymanopt/tools/autodiff/_autograd.py#L26)
        
- manifolds

    - stiefel
        - [todo @120 (simplify expressions)](./pymanopt/manifolds/stiefel.py#L120) Not sure if this is really possible though...

    - pairmean is required by nelder_mead hence needs to be included in manifold, fixed_rank, grassmann, stiefel. 
      check whether tangent method is redundant
    - Implement tests for manifolds

- solvers

    - conjugate_gradient
        - [todo @90](./pymanopt/solvers/conjugate_gradient.py#L90), [133](./pymanopt/solvers/conjugate_gradient.py#L133) (implement precondition)

    - linesearch
        - [todo @11 (allow to set parameters)](./pymanopt/solvers/linesearch.py#L11)

    - nelder_mead
        - [todo @55](./pymanopt/solvers/nelder_mead.py#L55), [57](./pymanopt/solvers/nelder_mead.py#L57), [59](./pymanopt/solvers/nelder_mead.py#L59) 
        (comment on solver arguments)

    - trust_regions
        - [todo @80 (implement value checks)](./pymanopt/solvers/trust_regions.py#L80)
        
    - Implement tests for solvers
    
- autodiff
    - add tensorflow support
