Quickstart
==========

Pymanopt is a modular toolbox and hence easy to use.
All of the automatic differentiation is done behind the scenes so that the
amount of setup the user needs to do is minimal.
Usually only the following steps are required:

#. Instantiate a manifold :math:`\manM` from the :mod:`pymanopt.manifolds`
   package to optimize over.
#. Define a cost function :math:`f:\manM \to \R` to minimize using one of the
   backend decorators defined in :mod:`pymanopt.function`.
#. Create a :class:`pymanopt.Problem` instance tying the
   optimization problem together.
#. Instantiate a Pymanopt optimizer from :mod:`pymanopt.optimizers` and run it
   on the problem instance.

Installation
------------

Pymanopt is compatible with Python 3.8+, and depends on NumPy and SciPy.
Additionally, to use Pymanopt's built-in automatic differentiation, which we
strongly recommend, you need to setup your cost functions using either
`Autograd <https://github.com/HIPS/autograd>`_,
`JAX <https://jax.readthedocs.io/en/latest/>`_,
`TensorFlow <https://www.tensorflow.org>`_ or
`PyTorch <http://www.pytorch.org/>`_.
If you are unfamiliar with these packages and you are unsure which to go for,
we suggest to start with Autograd.
Autograd wraps thinly around NumPy, and is very simple to use, particularly if
you're already familiar with NumPy.
To get the latest version of Pymanopt, install it via `pip` by specifying the
backend(s) you want to use among `autograd`, `jax`, `torch` and `tensorflow`,
separated by a comma. For example:

.. code-block:: bash

    $ pip install "pymanopt[autograd]" # or [jax], [tensorflow], [torch]

A Simple Example
----------------

As a simple illustrative example, we consider the problem of estimating the
dominant eigenvector of a real symmetric matrix :math:`\vmA \in \R^{n \times
n}`.
As is well known, a dominant eigenvector of a matrix :math:`\vmA` is any vector
:math:`\opt{\vmx}` that maximizes the Rayleigh quotient

.. math::

    \begin{align*}
        f(\vmx) &= \frac{\inner{\vmx}{\vmA\vmx}}{\inner{\vmx}{\vmx}}
    \end{align*}

with :math:`\inner{\cdot}{\cdot}` denoting the canonical inner product on
:math:`\R^n`.
The value of :math:`f` at :math:`\opt{\vmx}` coincides with the largest
eigenvalue of :math:`\vmA`.
Clearly :math:`f` is scale-invariant since :math:`f(\vmx) = f(\alpha\vmx)` for
any :math:`\alpha \neq 0`.
Hence one may reframe the dominant eigenvector problem as the minimization
problem

.. math::

    \begin{align*}
        \opt{\vmx} = \argmin_{\vmx \in \sphere^{n-1}}\inner{-\vmx}{\vmA\vmx}
    \end{align*}

with :math:`\sphere^{n-1}` denoting the set of all unit-norm vectors in
:math:`\R^n`: the sphere manifold of dimension :math:`n-1`.

The following is a minimal working example of how to solve the above problem
using Pymanopt for a random symmetric matrix.
As indicated in the introduction above, we follow four simple steps: we
instantiate the manifold, create the cost function (using Autograd in this
case), define a problem instance which we pass the manifold and the cost
function, and run the minimization problem using one of the available
optimizers.

.. literalinclude:: ../quickstart.py
   :language: python

Running this example will produce (something like) the following:

.. code-block:: none

    Optimizing...
    Iteration    Cost                       Gradient norm
    ---------    -----------------------    --------------
       1         -1.2503690275924836e-01    7.05048162e-01
       2         -1.0431116420457707e+00    1.51569863e+00
       3         -1.2458243811321195e+00    1.37636667e+00
       4         -1.6286858265437518e+00    2.35101411e-01
       5         -1.6313904350787611e+00    1.82139847e-01
       6         -1.6349496764006330e+00    6.23255354e-02
       7         -1.6350482449558597e+00    5.54214374e-02
       8         -1.6353381478158906e+00    2.60793677e-02
       9         -1.6353731802404177e+00    1.97764049e-02
      10         -1.6354097505834411e+00    9.45411176e-03
      11         -1.6354154571432260e+00    6.50360873e-03
      12         -1.6354205179079946e+00    7.35994761e-04
      13         -1.6354205808170219e+00    1.50461572e-04
      14         -1.6354205830464705e+00    6.51456482e-05
      15         -1.6354205835572466e+00    5.44513086e-06
      16         -1.6354205835607958e+00    6.01426462e-07
    Terminated - min grad norm reached after 16 iterations, 0.40 seconds.

    Dominant eigenvector: [ 0.48812905  0.6259872  -0.60816944]
    Pymanopt solution: [ 0.48812918  0.62598704 -0.60816949]

Note that depending on the random seed used in the example,
the direction of the "true" dominant eigenvector and the
solution found by Pymanopt can differ.
Eigenvectors are not unique since every eigenpair :math:`(\lambda, \vmv)` still
satisfies the eigenvalue equation :math:`\vmA \vmv = \lambda \vmv` if
:math:`\vmv` is replaced by :math:`\alpha \vmv` for some :math:`\alpha \in \R
\setminus \set{0}`.
That is, the dominant eigenvector is only unique up to multiplication by a
nonzero constant; the zero vector is trivially considered *not* an eigenvector.

The example above constitutes the conceivably simplest demonstration of
Pymanopt.
For more interesting examples we refer to the `examples
<https://github.com/pymanopt/pymanopt/tree/master/examples>`_ in Pymanopt's
github repository.
Moreover, `this notebook <examples/notebooks/mixture_of_gaussians.ipynb>`_
demonstrates a more involved application of Riemannian optimization using
Pymanopt in the context of inference in Gaussian mixture models.
