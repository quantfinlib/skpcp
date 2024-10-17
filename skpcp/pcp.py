"""Implementation of Principal Component Pursuit (PCP)."""

from typing import Optional

import numpy as np 
from sklearn.base import BaseEstimator, TransformerMixin


def shrinkage_operator(X: np.ndarray, tau: float) -> np.ndarray:
    """Apply Shrinkage operator to a matrix X with threshold tau.

    .. math::
        S_{\\tau}(X) = \\text{sign}(X) \\cdot \\max(\\abs(X) - \\tau, 0)

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
    return np.sign(X) * np.maximum(np.abs(X) - tau, 0)


def svd_operator(X: np.ndarray, tau: float) -> np.ndarray:
    """Apply Singular Value Thresholding operator to a matrix X with threshold tau.

    .. math::
        D_{\\tau}(X) = U \\cdot \\text{shrinkage_operator}(S, \\tau) \\cdot V^T

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
    return U @ np.diag(shrinkage_operator(S, tau)) @ Vt


def l1_norm(X: np.ndarray) -> float:
    """Compute the L1 norm of a matrix.

    .. math::
        \\norm{X}_1 = \\sum_{i,j} \\abs{X_{ij}}

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
    """Compute the Frobenius norm of a matrix.

    .. math::
        \\norm{X}_F = \\sqrt{\\sum_{i,j} X_{ij}^2}

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    
    Returns
    -------
    float
        Frobenius norm of the matrix.
    """
    return np.linalg.norm(X, ord='fro')


def _pcp(X: np.ndarray, lambd: float, mu: Optional[float] = None, max_iter: int = 100, tol: float = 1e-7) -> tuple:
    """Apply Principal Component Pursuit (PCP) to decompose a matrix into low-rank and sparse components.

    Parameters
    ----------
    X : np.ndarray
        Input matrix.
    lambd : Optional[float]
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
        Low-rank matrix (L), sparse matrix (S).
    """
    # Initialize mu if not provided
    if mu is None:
        m, n = X.shape
        mu = m * n / (4 * l1_norm(X))
    
    if lambd is None:
        lambd = 1 / np.sqrt(max(X.shape))

    # Initialize variables: L (low-rank), S (sparse), Y (Lagrange multiplier)
    L = np.zeros_like(X)
    S = np.zeros_like(X)
    Y = np.zeros_like(X)
    for _ in range(max_iter):
        # Step 1: Update L (low-rank matrix)
        L_new = svd_operator(X - S + (1/mu) * Y, 1/mu)
        # Step 2: Update S (sparse matrix)
        S_new = shrinkage_operator(X - L_new + (1/mu) * Y, lambd/mu)
        # Step 3: Update Y (Lagrange multiplier)
        Y_new = Y + mu * (X - L_new - S_new)
        # Check convergence
        if frobenius_norm(X - L_new - S_new) / (frobenius_norm(X) + 1e-8) < tol:
            break
        # Update variables
        L, S, Y = L_new, S_new, Y_new

    return L, S


class PCP(BaseEstimator, TransformerMixin):
    """Principal Component Pursuit (PCP) for matrix decomposition.

    Parameters
    ----------
    lambd : float
        Regularization parameter for the sparse component.
    mu : Optional[float]
        Augmented Lagrange multiplier parameter. If None, it's set automatically.
    max_iter : int, optional
        Maximum number of iterations, by default 100.
    tol : float, optional
        Tolerance for convergence, by default 1e-7.
    """
    def __init__(self, lambd: Optional[float] = None, mu: Optional[float] = None, max_iter: int = 100, tol: float = 1e-7):
        self.lambd = lambd
        self.mu = mu
        self.max_iter = max_iter
        self.tol = tol

    def fit_transform(self, X: np.ndarray) -> tuple:
        """Fit the model with the input matrix and return the low-rank and sparse components.

        Parameters
        ----------
        X : np.ndarray
            Input matrix.
        
        Returns
        -------
        tuple
            Low-rank matrix (L), sparse matrix (S).
        """
        return _pcp(X, self.lambd, self.mu, self.max_iter, self.tol)
    
    def transform(self, X: np.ndarray) -> tuple:
        """Transform the input matrix and return the low-rank and sparse components.

        Parameters
        ----------
        X : np.ndarray
            Input matrix.
        
        Returns
        -------
        tuple
            Low-rank matrix (L), sparse matrix (S).
        """
        return self.fit_transform(X)
    
    def fit(self, X: np.ndarray):
        """Fit the model with the input matrix.

        Parameters
        ----------
        X : np.ndarray
            Input matrix.
        """
        self.fit_transform(X)
        return self