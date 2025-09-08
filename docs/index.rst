skpcp
===========

SkPCP: a package for robust principal component analysis or *principal component pursuit*
built on top of `RobustPCA <https://dl.acm.org/doi/10.1145/1970392.1970395/>`

- The basic idea is to decompose a data matrix D into a low-rank matrix L and a sparse matrix S, i.e., D = L + S.
- Principal component pursuit aims at recovering L and S by solving the convex optimization problem:

  min $||L||_* + λ||S||_1$ subject to $D = L + S$, where the parameter $\lambda > 0$ 
  is a regularization parameter that balances the trade-off between the rank and sparsity.



Installation
------------------------
.. code-block:: bash

    pip install skpcp


.. toctree::
   :caption: Getting started
   :glob:
   :maxdepth: 1

   getting_started

.. toctree::
   :caption: Examples
   :glob:

   examples/examples

.. toctree::
   :caption: API
   :maxdepth: 1
   :glob:

   api/pcp

.. toctree::
   :maxdepth: 1
   :caption: Miscellaneous
   
   faq