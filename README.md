# pymanopt

[![Join the chat at https://gitter.im/j-towns/pymanopt](https://badges.gitter.im/j-towns/pymanopt.svg)](https://gitter.im/j-towns/pymanopt?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://travis-ci.org/j-towns/pymanopt.svg?branch=master)](https://travis-ci.org/j-towns/pymanopt)
[![Coverage Status](https://coveralls.io/repos/github/j-towns/pymanopt/badge.svg?branch=master)](https://coveralls.io/github/j-towns/pymanopt?branch=master)

Python toolbox for manifold optimization that computes gradients and hessians automatically.

A documentation will be made available in the near future at https://pymanopt.github.io.

Currently supports [theano](http://deeplearning.net/software/theano/) and [autograd](https://github.com/HIPS/autograd) as autodiff backends.

Builds upon the MATLAB package [Manopt](http://manopt.org/) but is otherwise independent of it.


## Manopt feature implementation
### Manifolds

| Manifold | Implemented |
| ------------- |:-----------:|
| Euclidean | Partially |
| Sphere | Partially |
| Stiefel | Partially |
| Grassmann | Partially |
| Symmetric positive semidefinite,<br>fixed-rank (complex) | Partially |
| Symmetric positive semidefinite,<br>fixed-rank with unit diagonal | Partially |
| Oblique manifold | Partially |

### Solvers

| Solver | Type | Implemented |
| ------ | :--: | :---------: |
| Steepest-descent | First-order | Partially |
| Conjugate-gradient | First-order | Partially |
| Trust-regions | Second-order | Partially |
| Particle swarm (PSO) | Derivative-free | Partially |
| Nelder-Mead | Derivative-free | Partially |

## Installation
### Dependencies
This package depends on python 2.7.*, numpy, scipy and (theano or autograd).
Instructions for installing numpy, scipy and theano on different operating systems can be found
[here](http://deeplearning.net/software/theano/install.html), for installing autograd [here](https://github.com/HIPS/autograd#how-to-install).

### Installing pymanopt
You can install pymanopt with the following command:
```
pip install --user git+https://github.com/j-towns/pymanopt.git
```

## Basic usage
To do optimization with pymanopt, you will need to create a manifold object, a
solver object, and a cost function. Classes of manifolds and solvers are
provided with pymanopt.
In case you want to make use of pymanopt's autodiff functionality, cost functions have to be set up using theano or in a autograd compatible fashion.
A tutorial on theano can be found
[here](http://deeplearning.net/software/theano/tutorial/index.html),
one on autograd [here](https://github.com/HIPS/autograd/blob/master/docs/tutorial.md).

### Example code
```python
import theano.tensor as T
import numpy as np

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pymanopt.manifolds import Stiefel

# ---------------------------------
# Define cost function using theano
# ---------------------------------
# Note, your cost function needs to have one (matrix) input and one (scalar) output.
X = T.matrix()

# Cost is the sum of all of the elements of the matrix X.
cost = T.sum(X)

# ---------------------------------
# Setup solver and manifold objects
# ---------------------------------
solver = SteepestDescent()
manifold = Stiefel(5, 2)

# --------------------
# Setup problem object
# --------------------
problem = Problem(man=manifold, cost=cost, arg=X)

# --------------------
# Perform optimization
# --------------------
# Currently the solve function takes the problem object as input.
Xopt = solver.solve(problem)

print(Xopt)
```
See [here](https://github.com/j-towns/pymanopt/tree/master/examples) for more
examples.
