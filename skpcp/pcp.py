# Copyright (c) 2025 Mohammadjavad Vakili. All rights reserved.

"""Implementation of Principal Component Pursuit (PCP)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data

if TYPE_CHECKING:
    from numpy.typing import ArrayLike

NUMERICAL_TOLERANCE = 1e-8


def shrinkage_operator(X: np.ndarray, tau: float) -> np.ndarray:
    r"""Apply Shrinkage operator to a matrix X with threshold tau.

    .. math::

        S_{\tau}(X) = \text{sign}(X) \cdot \max(|X| - \tau, 0)

    where :math: `\text{sign}(X)` is the element-wise sign function, and
    :math: `\max(|X| - \tau, 0)` is the element-wise maximum between :math: `|X| - \tau` and :math: `0`.


    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    tau : float
        Threshold value.

    Returns
    -------
    np.ndarray
        Shrunk matrix.

    """
    result = np.sign(X) * np.maximum(np.abs(X) - tau, 0)
    # Set values below numerical tolerance to exactly zero
    result[np.abs(result) < NUMERICAL_TOLERANCE] = 0
    return result


def svd_operator(X: np.ndarray, tau: float) -> np.ndarray:
    r"""Apply Singular Value Thresholding operator to a matrix X with threshold tau.

    $$D_{\\tau}(X) = U \\cdot \\text{shrinkage_operator}(S, \\tau) \\cdot V^T $$

    where :math:`X = U S V^T` is the Singular Value Decomposition (SVD) of :math:`X`.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    tau : float
        Threshold value.

    Returns
    -------
    np.ndarray
        Shrunk SVD.

    """
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    return U @ np.diag(shrinkage_operator(X=S, tau=tau)) @ Vt


def l1_norm(X: np.ndarray) -> float:
    r"""Compute the L1 norm of a matrix.

    .. math::

        \|X\|_1 = \sum_{i,j} |X_{ij}|

    Parameters
    ----------
    X : np.ndarray
        Input matrix.

    Returns
    -------
    float
        L1 norm of the matrix.

    """
    return np.sum(np.abs(X))


def frobenius_norm(X: np.ndarray) -> float:
    r"""Compute the Frobenius norm of a matrix.

    .. math::

        \|X\|_F = \sqrt{\sum_{i,j} X_{ij}^2}

    Parameters
    ----------
    X : np.ndarray
        Input matrix.

    Returns
    -------
    float
        Frobenius norm of the matrix.

    """
    return float(np.linalg.norm(X, ord="fro"))


def pcp(
    X: np.ndarray,  # noqa: N803
    alpha: float | None = None,
    mu: float | None = None,
    max_iter: int = 100,
    tol: float = 1e-7,
) -> tuple:
    """Apply Principal Component Pursuit (PCP) to decompose a matrix into low-rank and sparse components.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    alpha : Optional[float]
        Regularization parameter for the sparse component.
    mu : Optional[float]
        Augmented Lagrange multiplier parameter. If None, it's set automatically.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-7.

    Returns
    -------
    tuple
        Low-rank matrix (L), sparse matrix (S), number of iterations (`n_iter_`).

    """
    # Initialize mu if not provided
    if mu is None:
        m, n = X.shape
        mu = m * n / (4 * l1_norm(X))
    # Initialize alpha if not provided
    if alpha is None:
        alpha = 1 / np.sqrt(max(X.shape))

    # Initialize variables: L (low-rank), S (sparse), Y (Lagrange multiplier)
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    n_iter_ = 0
    for _ in range(max_iter):
        # Step 1: Update L (low-rank matrix)
        L_new = svd_operator(X=X - S + (1. / mu) * Y, tau=1. / mu)
        # Step 2: Update S (sparse matrix)
        S_new = shrinkage_operator(X - L_new + (1 / mu) * Y, tau=alpha / mu)
        # Step 3: Update Y (Lagrange multiplier)
        Y_new = Y + mu * (X - L_new - S_new)
        # Update variables
        L, S, Y = L_new, S_new, Y_new
        n_iter_ += 1
        # Check convergence
        if frobenius_norm(X - L_new - S_new) / (frobenius_norm(X) + NUMERICAL_TOLERANCE) < tol:
            break
    return L, S, n_iter_


class PCP(TransformerMixin, BaseEstimator):
    """Principal Component Pursuit (PCP) for matrix decomposition.

    PCP decomposes an input matrix into low-rank and sparse components using
    augmented Lagrange multiplier method.

    Parameters
    ----------
    alpha : float, default=None
        Regularization parameter for the sparse component. If None, it's set
        automatically as 1/sqrt(max(n_features, n_samples)).
    mu : float, default=None
        Augmented Lagrange multiplier parameter. If None, it's set automatically
        as n_features * n_samples / (4 * L1_norm(X)).
    max_iter : int, default=100
        Maximum number of iterations for the optimization algorithm.
    tol : float, default=1e-7
        Tolerance for convergence. Algorithm stops when the relative error
        is below this threshold.

    Attributes
    ----------
    low_rank_ : ndarray of shape (n_samples, n_features)
        The low-rank component from the last decomposition.
    sparse_ : ndarray of shape (n_samples, n_features)
        The sparse component from the last decomposition.
    n_iter_ : int
        Number of iterations run in the fit.

    """

    def __init__(  # noqa: D107
        self,
        alpha: float | None = None,
        mu: float | None = None,
        max_iter: int = 100,
        tol: float = 1e-7,
    ) -> None:
        self.alpha = alpha
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol

    def __sklearn_tags__(self) -> dict[str, Any]:  # noqa: PLW3201
        """Return sklearn estimator tags for compatibility.

        Returns
        -------
        dict
            Dictionary of estimator tags.

        """
        tags = super().__sklearn_tags__()
        tags.target_tags.required = False
        tags.requires_fit = True
        tags.input_tags.two_d_array = True
        tags.input_tags.allow_nan = False
        tags.input_tags.positive_only = False
        tags.input_tags.sparse = False
        tags.no_validation = False
        return tags

    def fit(self, X: ArrayLike, y: Any = None) -> PCP:
        """Fit the PCP model to the input matrix.

        This method stores the input matrix dimensions and validates parameters,
        but does not perform the actual decomposition until transform is called.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input matrix to fit the model to.
        y : array-like, default=None
            Target values (ignored). This parameter exists for sklearn compatibility.

        Returns
        -------
        self : PCP
            Fitted estimator instance.

        """
        # Validate input and store sklearn-required attributes
        X = validate_data(
            self,
            X=X,
            y=None,
            reset=True,
        )
        self.low_rank_, self.sparse_, self.n_iter_ = pcp(
            X=X,
            alpha=self.alpha,
            mu=self.mu,
            max_iter=self.max_iter,
            tol=self.tol,
        )
        if self.alpha is None:
            self.alpha = 1 / np.sqrt(max(X.shape))
        if self.mu is None:
            m, n = X.shape
            self.mu = m * n / (4 * l1_norm(X))
        return self

    def transform(self, X: ArrayLike) -> np.ndarray:
        """Transform input matrix by returning the low-rank component.

        Only supports the exact same data used in fit.

        Parameters
        ----------
        X    : array-like of shape (n_samples, n_features)
            Input matrix to transform.

        Returns
        -------
        x_transformed : ndarray of shape (n_samples, n_features)
            Low-rank component of the input matrix.

        """
        check_is_fitted(self, attributes=["low_rank_", "sparse_"])
        return self.low_rank_
