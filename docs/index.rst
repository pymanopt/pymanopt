Pymanopt
========

Pymanopt is a Python toolbox for optimization on Riemannian manifolds.
The project aims to lower the barrier for users wishing to use state of the art
techniques for optimization on manifolds by relying on automatic
differentiation for computing gradients and Hessians, saving users time and
saving them from potential calculation and implementation errors.

Pymanopt is modular and hence easy to use. All of the automatic differentiation
is done behind the scenes, so that the amount of setup the user needs to do is
minimal. Usually only the following steps are required:

#. Instantiate a manifold :math:`\man` to optimize over
#. Define a cost function :math:`f:\man \to \R` to minimise
#. Instantiate a Pymanopt solver

Experimenting with optimization on manifolds is simple with Pymanopt.
The minimum working example below demonstrates this.
Please refer to `this example <examples/notebooks/mixture_of_gaussians.ipynb>`_
for a more involved example of Riemannian optimization using Pymanopt for
inference in MoG models.

We encourage users and developers to report problems, request
features, ask for help, or leave general comments either on
`github <https://github.com/pymanopt/pymanopt>`_ or
`gitter <https://gitter.im/pymanopt/pymanopt>`_.

Please refer to our `contributing guide <CONTRIBUTING.md>`_ if you wish to
extend Pymanopt's functionality and/or contribute to the project.
Pymanopt is distributed under the open source `3-clause BSD license
<https://github.com/pymanopt/pymanopt/blob/master/LICENSE>`_.
Please cite `this JMLR paper <http://jmlr.org/papers/v17/16-177.html>`_ if you
publish work using Pymanopt (`BibTeX
<http://jmlr.org/papers/v17/16-177.bib>`_).

Quick install
-------------

.. code-block:: bash

    pip install pymanopt

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   Examples <https://github.com/pymanopt/pymanopt/tree/master/examples>
   api-reference.rst
   CONTRIBUTING.md

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   examples/notebooks/mixture_of_gaussians.ipynb
