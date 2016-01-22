# pymanopt
[![Build Status](https://travis-ci.org/j-towns/pymanopt.svg?branch=master)](https://travis-ci.org/j-towns/pymanopt)

Python port of the MATLAB package Manopt, for manifold optimization using Theano for automatic differentiation.

This project is independent from the Manopt project.

http://www.manopt.org

http://deeplearning.net/software/theano/

## Manopt feature implementation
### Manifolds

| Manifold | Implemented |
| ------------- |:-----------:|
| Sphere | Partially |
| Stiefel | Partially |
| Grassmann | Partially |
| Symmetric positive semidefinite, fixed-rank | Partially |

### Solvers

| Solver | Implemented |
| ------ |:-------------:|
| Steepest-descent| Partially |
| Conjugate-gradient | Partially |
| Trust-regions | Partially |
| Particle swarm (PSO) | Partially |
| Nelder-Mead | Partially |

## Installation
### Dependencies
This package depends on python 2.7.*, numpy, scipy and theano. Instructions for installing numpy, scipy and Theano on different operating systems can be found [here](http://deeplearning.net/software/theano/install.html).

### Installing pymanopt
You can install pymanopt with the following command:
```
pip install --user git+https://github.com/j-towns/pymanopt.git
```

## Basic usage
To do optimization with pymanopt, you will need to create a manifold object, a solver object, and a cost function. Classes of manifolds and solvers are provided with pymanopt. Cost functions have to be set up using Theano. A tutorial on Theano can be found [here](http://deeplearning.net/software/theano/tutorial/index.html).

### Example code
```python
import theano.tensor as T
import numpy as np

from pymanopt import Problem
from pymanopt.solvers import SteepestDescent
from pymanopt.manifolds import Stiefel

# ---------------------------------
# Define cost function using Theano
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
problem = Problem(man=manifold, theano_cost=cost, theano_arg=X)

# --------------------
# Perform optimization
# --------------------
# Currently the solve function requires three inputs: the cost and input variable
# (both defined using theano) and the manifold to optimise over.
Xopt = solver.solve(problem)

print Xopt
```
See [here](https://github.com/j-towns/pymanopt/tree/master/examples) for more examples.
