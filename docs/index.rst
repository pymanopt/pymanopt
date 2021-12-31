Pymanopt
========

Pymanopt is a Python toolbox for optimization on Riemannian manifolds.
The project aims to lower the barrier for users wishing to use state-of-the-art
techniques for optimization on manifolds by leveraging automatic
differentiation for computing gradients and Hessians, thus saving users time
and reducing the potential of calculation and implementation errors.

We encourage users and developers to report problems, request
features, ask for help, or leave general comments either on
`github <https://github.com/pymanopt/pymanopt>`_ or
`gitter <https://gitter.im/pymanopt/pymanopt>`_.
Please refer to our `contributing guide <CONTRIBUTING.md>`_ if you wish to
extend Pymanopt's functionality and/or contribute to the project.

Pymanopt is distributed under the open source `3-clause BSD license
<https://github.com/pymanopt/pymanopt/blob/master/LICENSE>`_.
Please cite `this JMLR paper <http://jmlr.org/papers/v17/16-177.html>`_ if you
publish work using Pymanopt:

.. code-block::

    @article{JMLR:v17:16-177,
        author = {James Townsend and Niklas Koep and Sebastian Weichwald},
        journal = {Journal of Machine Learning Research},
        number = {137},
        pages = {1–5},
        title = {Pymanopt: A Python Toolbox for Optimization on Manifolds using Automatic Differentiation},
        url = {http://jmlr.org/papers/v17/16-177.html},
        volume = {17},
        year = {2016}
    }

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   quickstart.rst
   Examples <https://github.com/pymanopt/pymanopt/tree/master/examples>
   api-reference.rst
   CONTRIBUTING.md

.. toctree::
   :maxdepth: 1
   :caption: Notebooks

   examples/notebooks/mixture_of_gaussians.ipynb
