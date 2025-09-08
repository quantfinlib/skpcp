import numpy as np
import pytest
from scipy import sparse

from skpcp.pcp import NUMERICAL_TOLERANCE, PCP, frobenius_norm, l1_norm, pcp, shrinkage_operator, svd_operator

RNG = np.random.default_rng(42)
SINGULAR_VALUES_RANK_3 = [.1, 1., 10.]
SINGULAR_VALUES_RANK_2 = [.1, 1., 0.]


@pytest.fixture(scope="module")
def matrix_fixture(request):
    kind = request.param
    x, y = np.cos(np.pi / 4), np.sin(np.pi / 4)
    if kind == "trivial_rank_3":
        S = np.diag(SINGULAR_VALUES_RANK_3)
        U = np.eye(3)
        V = np.eye(3)
        return U @ S @ V.T
    elif kind == "non_trivial_rank_3":
        S = np.diag(SINGULAR_VALUES_RANK_3)
        U = np.array([[x, 0, y], [0, 1, 0], [-y, 0, x]])
        V = np.array([[1, 0, 0], [0, x, -y], [0, y, x]])
        return U @ S @ V.T
    elif kind == "trivial_rank_2":
        S = np.diag(SINGULAR_VALUES_RANK_2)
        U = np.eye(3)
        V = np.eye(3)
        return U @ S @ V.T
    elif kind == "non_trivial_rank_2":
        S = np.diag(SINGULAR_VALUES_RANK_2)
        U = np.array([[x, 0, y], [0, 1, 0], [-y, 0, x]])
        V = np.array([[1, 0, 0], [0, x, -y], [0, y, x]])
        return U @ S @ V.T
    elif kind == "rank_3_with_noise":
        S = np.diag(SINGULAR_VALUES_RANK_3)
        U = np.array([[x, 0, y], [0, 1, 0], [-y, 0, x]])
        V = np.array([[1, 0, 0], [0, x, -y], [0, y, x]])
        base = U @ S @ V.T
        noise = RNG.normal(0, 0.01, base.shape)
        return base + noise
    else:
        msg = f"Unknown matrix type: {kind}"
        raise ValueError(msg)




def test_shrinkage_operator_1d_vector() -> None:
    """Test the shrinkage operator."""
    S = np.array([1.0, 0.5, 0.2])

    tau = 0.3
    expected = np.array([0.7, 0.2, 0.0])
    result = shrinkage_operator(S, tau)
    np.testing.assert_array_almost_equal(result, expected)

    tau = 0.0
    expected = S.copy()
    result = shrinkage_operator(S, tau)
    np.testing.assert_array_almost_equal(result, expected)

    tau = 1.0
    expected = np.array([0.0, 0.0, 0.0])
    result = shrinkage_operator(S, tau)
    np.testing.assert_array_almost_equal(result, expected)

    S *= -1.  # Test with negative values
    tau = 0.3
    expected = -1. * np.array([0.7, 0.2, 0.0])
    result = shrinkage_operator(S, tau)
    np.testing.assert_array_almost_equal(result, expected)



@pytest.fixture(scope="module")
def rank_3_matrix_with_noise(non_trivial_rank_3_matrix: np.ndarray) -> np.ndarray:
    """Generate a rank 3 matrix with noise."""
    base_matrix = non_trivial_rank_3_matrix
    noise = RNG.normal(0, 0.01, base_matrix.shape)
    return base_matrix + noise


@pytest.mark.parametrize(
    "matrix_fixture",
    [
        "trivial_rank_3",
        "non_trivial_rank_3",
        "trivial_rank_2",
        "non_trivial_rank_2",
    ],
    indirect=True,
)
@pytest.mark.parametrize("tau", [0.1, 0.5, 1.0])
def test_svd_operator(matrix_fixture: np.ndarray, tau: float) -> None:
    """Test the SVD operator."""
    shrunk_matrix = svd_operator(matrix_fixture, tau)
    rank = np.linalg.matrix_rank(shrunk_matrix)
    original_rank = np.linalg.matrix_rank(matrix_fixture)
    assert rank <= original_rank, f"Expected rank <= {original_rank}, got {rank}"
    if original_rank == 3:
        expected_rank = np.sum(np.array(SINGULAR_VALUES_RANK_3) > tau)
    else:
        expected_rank = np.sum(np.array(SINGULAR_VALUES_RANK_2) > tau)
    assert rank == expected_rank, f"Expected rank after shrinking {expected_rank}, got {rank}"


def test_l1_norm() -> None:
    """Test the L1 norm function."""
    X = np.array([[1, -2, 3], [-4, 5, -6]])
    expected = 21.0
    result = l1_norm(X)
    assert np.isclose(result, expected), f"Expected L1 norm {expected}, got {result}"


def test_frobenius_norm() -> None:
    """Test the Frobenius norm function."""
    X = np.array([[1, -2, 3], [-4, 5, -6]])
    expected = np.sqrt(1 + 4 + 9 + 16 + 25 + 36)
    result = frobenius_norm(X)
    assert np.isclose(result, expected), f"Expected Frobenius norm {expected}, got {result}"


@pytest.mark.parametrize(
    "matrix_fixture",
    [
        "trivial_rank_3",
        "non_trivial_rank_3",
        "trivial_rank_2",
        "non_trivial_rank_2",
        "rank_3_with_noise",
    ],
    indirect=True,
)
@pytest.mark.parametrize("max_iter", [100, 500, 1000])
@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-8])
def test_pcp(matrix_fixture: np.ndarray, max_iter: int, tol: float) -> None:
    """Test the PCP algorithm on a rank 3 matrix with noise."""
    input_data = matrix_fixture
    low_rank, sparse, n_iter = pcp(input_data, max_iter=max_iter, tol=tol)
    reconstruction = low_rank + sparse
    error = (frobenius_norm(input_data - reconstruction) /
             (frobenius_norm(input_data) + NUMERICAL_TOLERANCE))
    convergence_criteria = error <= tol or n_iter <= max_iter
    msg = f"PCP did not converge within {max_iter} iterations or tolerance {tol}"
    assert convergence_criteria, msg
    low_rank_rank = np.linalg.matrix_rank(low_rank)
    input_data_rank = np.linalg.matrix_rank(input_data)
    msg = f"Expected low-rank component to have rank <= {input_data_rank}, got {low_rank_rank}"
    assert low_rank_rank <= input_data_rank, msg
    sparse_rank = np.linalg.matrix_rank(sparse)
    msg = f"Expected sparse component to have rank <= {input_data_rank}, got {sparse_rank}"
    assert sparse_rank <= input_data_rank, msg


@pytest.mark.parametrize(
    "matrix_fixture",
    [
        "trivial_rank_3",
        "non_trivial_rank_3",
        "trivial_rank_2",
        "non_trivial_rank_2",
        "rank_3_with_noise",
    ],
    indirect=True,
)
@pytest.mark.parametrize("max_iter", [100, 500, 1000])
@pytest.mark.parametrize("tol", [1e-4, 1e-6, 1e-8])
def test_pcp_class(matrix_fixture: np.ndarray, max_iter: int, tol: float) -> None:
    """Test the PCP class."""
    model = PCP(max_iter=max_iter, tol=tol)
    model.fit(matrix_fixture)
    low_rank = model.low_rank_
    sparse = model.sparse_
    n_iter = model.n_iter_
    reconstruction = low_rank + sparse
    error = (frobenius_norm(matrix_fixture - reconstruction) /
             (frobenius_norm(matrix_fixture) + NUMERICAL_TOLERANCE))
    convergence_criteria = error <= tol or n_iter <= max_iter
    msg = f"PCP did not converge within {max_iter} iterations or tolerance {tol}"
    assert convergence_criteria, msg
    low_rank_rank = np.linalg.matrix_rank(low_rank)
    input_data_rank = np.linalg.matrix_rank(matrix_fixture)
    msg = f"Expected low-rank component to have rank <= {input_data_rank}, got {low_rank_rank}"
    assert low_rank_rank <= input_data_rank, msg
    sparse_rank = np.linalg.matrix_rank(sparse)
    msg = f"Expected sparse component to have rank <= {input_data_rank}, got {sparse_rank}"
    assert sparse_rank <= input_data_rank, msg


def test_pcp_class_invalid_input_data() -> None:
    """Test PCP class with invalid input."""
    model = PCP()
    X = np.array([1, 2, 3])  # 1D array instead of 2D
    with pytest.raises(ValueError, match=r"Expected 2D array, got 1D array instead"):
        model.fit(X)
    X = np.array([[[1, 2], [3, 4]]])  # 3D array instead of 2D
    with pytest.raises(ValueError, match=r"Found array with dim 3, while dim <= 2 is required by PCP."):
        model.fit(X)
    # sparse sklearn input
    X = sparse.csr_matrix([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match=r"Sparse data was passed for X, but dense data is required."):
        model.fit(X)


def test_pcp_class_behavior_before_fit() -> None:
    """Test PCP class behavior before calling fit."""
    model = PCP()
    X = np.array([[1, 2], [3, 4]])
    with pytest.raises(
        AttributeError,
        match=r"This PCP instance is not fitted yet.",
    ):
        model.transform(X)
    with pytest.raises(
        AttributeError,
        match=r"'PCP' object has no attribute 'low_rank_'",
    ):
        _ = model.low_rank_
    with pytest.raises(
        AttributeError,
        match=r"'PCP' object has no attribute 'sparse_'",
    ):
        _ = model.sparse_


def test_pcp_class_fit_transform():
    """Test the fit_transform method of the PCP class."""
    model = PCP()
    X = np.array([[1, 2], [3, 4]])
    model.fit(X)
    transformed = model.transform(X)
    assert transformed.shape == X.shape, f"Expected shape {X.shape}, got {transformed.shape}"
    assert np.array_equal(transformed, model.low_rank_), "Transform output does not match low_rank_ attribute"
    # Check that fit_transform gives the same result as fit followed by transform
    transformed_again = model.fit_transform(X)
    msg = "fit_transform output does not match fit followed by transform output"
    assert np.array_equal(transformed, transformed_again), msg


@pytest.mark.parametrize(
    "matrix_fixture",
    [
        "trivial_rank_3",
        "non_trivial_rank_3",
        "trivial_rank_2",
        "non_trivial_rank_2",
        "rank_3_with_noise",
    ],
    indirect=True,
)
def test_alpha_mu_initialization(matrix_fixture: np.ndarray) -> None:
    """Test automatic initialization of alpha and mu in PCP class."""
    model = PCP(alpha=None, mu=None)
    assert model.alpha is None, "Expected alpha to be None before fit"
    assert model.mu is None, "Expected mu to be None before fit"
    model.fit(matrix_fixture)
    expected_alpha = 1 / np.sqrt(max(matrix_fixture.shape))
    m, n = matrix_fixture.shape
    expected_mu = m * n / (4 * l1_norm(matrix_fixture))
    assert np.isclose(model.alpha, expected_alpha), f"Expected alpha {expected_alpha}, got {model.alpha}"
    assert np.isclose(model.mu, expected_mu), f"Expected mu {expected_mu}, got {model.mu}"
