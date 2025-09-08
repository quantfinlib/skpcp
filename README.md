## skpcp

Robust principal component analysis via Principal Component Pursuit (PCP) with scikit-learn transformer interface. 

[![codecov](https://codecov.io/gh/quantfinlib/skpcp/graph/badge.svg?token=ZUZEM2WENL)](https://codecov.io/gh/quantfinlib/skpcp)
[![Tests](https://github.com/quantfinlib/skpcp/actions/workflows/test.yml/badge.svg)](https://github.com/quantfinlib/skpcp/actions/workflows/test.yml)
[![Doc Build](https://github.com/quantfinlib/skpcp/actions/workflows/gh-pages.yml/badge.svg)](https://github.com/quantfinlib/skpcp/actions/workflows/docs.yml)


### Installation

```bash
pip install skpcp
```

### Getting Started

Principal Component Pursuit (PCP) is a method for decomposing a data matrix `X` into a low-rank component `L` and a sparse component `S`, i.e., `X = L + S`. The `skpcp` package provides an implementation of PCP with a scikit-learn compatible transformer interface.

At its core the algorithm solves the following optimization problem $$ \min_{L,S} \|L\|_* + \lambda \|S\|_1 \quad \text{s.t.} \quad X = L + S $$ where $\|L\|_*$ is the nuclear norm (sum of singular values) of `L`, $\|S\|_1$ is the element-wise $\ell_1$ norm of `S`, and $\lambda > 0$ is a regularization parameter that controls the trade-off between the low-rank and sparse components. In practice, the user does not need to set the value of $\lambda$, as it is automatically chosen based on the dimensions of the input data matrix `X`.
We refer the users to the original paper by Candes et al. (2011) for more details: [Robust Principal Component Analysis?](https://www.microsoft.com/en-us/research/wp-content/uploads/2009/12/RobustPCA.pdf).


```python

import numpy as np
from skpcp import PCP

# Generate synthetic data with low-rank and sparse components
RNG = np.random.default_rng(42)
n_samples, n_features, rank = 100, 50, 5
L = np.dot(RNG.normal(size=(n_samples, rank)), RNG.normal(size=(rank, n_features)))  # Low rank component
S = RNG.binomial(1, 0.1, size=(n_samples, n_features)) * RNG.normal(loc=0, scale=10, size=(n_samples, n_features))  # Sparse component
X = L + S

# Fit PCP model
pcp = PCP()
pcp.fit(X)
L_est = pcp.low_rank_  # Estimated low-rank component
S_est = pcp.sparse_  # Estimated sparse component

```
Alternatively you can use the `fit_transform` method to fit the model and obtain the low-rank component in one step:

```python
L_est = pcp.fit_transform(X)
```

Note that the `fit` method decomposes the input data matrix `X` into its low-rank component `L_est` and sparse component `S_est`.
The behavior of the `transform`method of `PCP` differs from that of a typical scikit-learn transformer, in that it accepts the same data matrix `X` that was used in `fit`. You cannot pass a new data matrix to `transform`, as the decomposition is specific to the input data used in `fit`.

Please see the [examples](https://quantfinlib.github.io/skpcp/examples/examples.html) and the [API reference](https://quantfinlib.github.io/skpcp/api/pcp.html) for more details.


### [Documentation](https://quantfinlib.github.io/skpcp/)

The documentation is supported by [Sphinx](https://www.sphinx-doc.org/en/master/) and it is hosted on [GitHub pages](https://quantfinlib.github.io/skpcp/). 

To build the HTML pages locally, first make sure you have installed the package with its documentation dependencies:

```bash
uv pip install -e .[docs]
```

then run the following:

```bash
sphinx-build docs docs/_build
```