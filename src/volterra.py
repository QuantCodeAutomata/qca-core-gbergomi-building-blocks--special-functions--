"""
Volterra integral covariance structure for the grey Bergomi model.

The unscaled Volterra process is

    V_tilde_t = integral_0^t (t - u)^{H - 0.5} dW_u,

where H is the Hurst roughness exponent (0 < H < 0.5).  Its covariance
structure admits a closed-form expression via the Gauss hypergeometric
function 2F1, enabling exact simulation through Cholesky decomposition of
the covariance matrix.

Public API
----------
volterra_covariance                : Covariance Cov(V_tilde_t, V_tilde_s).
build_volterra_covariance_matrix   : Covariance matrix over a time grid.
simulate_volterra_gaussian         : Exact Gaussian paths via Cholesky.
volterra_conditional_covariance    : Numerical conditional covariance integral.
"""

import numpy as np
from scipy.integrate import quad
from scipy.special import hyp2f1


# ---------------------------------------------------------------------------
# Scalar covariance formula
# ---------------------------------------------------------------------------


def volterra_covariance(t: float, s: float, H: float) -> float:
    """
    Covariance of the unscaled Volterra integrals V_tilde_t and V_tilde_s.

    For t <= s the closed-form expression is

        Cov(V_tilde_t, V_tilde_s) =
            t^{H+0.5} * s^{H-0.5} / (H + 0.5)
            * 2F1(-H + 0.5, 1, H + 1.5; t/s),

    where 2F1 is the Gauss hypergeometric function.  By symmetry the result
    for t > s is obtained by swapping arguments.  The special case t = s
    gives the variance Var(V_tilde_t) = t^{2H} / (2H).

    Parameters
    ----------
    t : float
        First time point (>= 0).
    s : float
        Second time point (>= 0).
    H : float
        Hurst exponent, 0 < H < 0.5.

    Returns
    -------
    float
        Cov(V_tilde_t, V_tilde_s).
    """
    if t < 0.0 or s < 0.0:
        raise ValueError(f"t and s must be non-negative, got t={t}, s={s}")
    if t == 0.0 or s == 0.0:
        return 0.0

    # Ensure t <= s (covariance is symmetric)
    if t > s:
        t, s = s, t

    ratio = t / s
    cov = (
        t ** (H + 0.5)
        * s ** (H - 0.5)
        / (H + 0.5)
        * hyp2f1(-H + 0.5, 1.0, H + 1.5, ratio)
    )
    return float(cov)


# ---------------------------------------------------------------------------
# Covariance matrix
# ---------------------------------------------------------------------------


def build_volterra_covariance_matrix(
    t_grid: np.ndarray,
    H: float,
) -> np.ndarray:
    """
    Build the full covariance matrix for V_tilde over a time grid.

    Uses broadcasting and scipy's vectorised ``hyp2f1`` to evaluate the
    closed-form covariance formula for all pairs (t_i, t_j) efficiently.

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of non-negative time points, shape ``(N,)``.
    H : float
        Hurst exponent, 0 < H < 0.5.

    Returns
    -------
    np.ndarray
        Symmetric positive-(semi)definite covariance matrix, shape ``(N, N)``.
        Entries corresponding to t = 0 are zero.
    """
    t = np.asarray(t_grid, dtype=float)
    N = len(t)

    # Build (N, N) grids: T[i,j] = t[i], S[i,j] = t[j]
    T, S = np.meshgrid(t, t, indexing="ij")  # shapes (N, N)
    T_min = np.minimum(T, S)
    T_max = np.maximum(T, S)

    # Ratio t_min / t_max, guarded against division by zero (t_max = 0)
    # np.where evaluates both branches eagerly, so we suppress the resulting
    # floating-point warnings from the "dead" branch computations.
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(T_max > 0.0, T_min / T_max, 0.0)

        # Evaluate hypergeometric on the full grid (vectorised)
        hf = hyp2f1(-H + 0.5, 1.0, H + 1.5, ratio)

        # Covariance: t_min^{H+0.5} * t_max^{H-0.5} / (H+0.5) * 2F1(...)
        # Guard 0^{H-0.5} (t_max=0 case; H-0.5 < 0 → 0^neg = inf) via mask.
        C = np.where(
            T_min > 0.0,
            T_min ** (H + 0.5) * T_max ** (H - 0.5) / (H + 0.5) * hf,
            0.0,
        )
    # Symmetrise to suppress floating-point asymmetry from hyp2f1
    C = 0.5 * (C + C.T)
    return C


# ---------------------------------------------------------------------------
# Exact Gaussian simulation via Cholesky
# ---------------------------------------------------------------------------


def simulate_volterra_gaussian(
    t_grid: np.ndarray,
    H: float,
    n_paths: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Simulate exact Gaussian paths of V_tilde via Cholesky decomposition.

    The covariance matrix is built from ``build_volterra_covariance_matrix``
    and Cholesky-factorised.  A small regularisation (1e-12 * I) is added to
    guard against near-singularity from floating-point rounding.

    If t_grid[0] == 0, V_tilde_0 = 0 is set deterministically for all paths;
    the Cholesky factor L corresponds to the non-zero sub-grid t_grid[1:].

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of non-negative time points, shape ``(N,)``.
    H : float
        Hurst exponent, 0 < H < 0.5.
    n_paths : int
        Number of independent sample paths.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    V_tilde : np.ndarray
        Simulated Volterra paths, shape ``(n_paths, N)``.
    L : np.ndarray
        Lower-triangular Cholesky factor of the covariance matrix for
        the non-zero sub-grid.  Suitable for reuse without re-factorising.
    """
    t = np.asarray(t_grid, dtype=float)

    # Split off the t=0 entry (V_tilde_0 = 0 deterministically, covariance=0)
    has_zero_start = t[0] == 0.0
    t_active = t[1:] if has_zero_start else t
    N_active = len(t_active)

    if N_active == 0:
        # Only t=0 in the grid — trivial
        L = np.zeros((0, 0))
        return np.zeros((n_paths, len(t))), L

    C = build_volterra_covariance_matrix(t_active, H)
    # Regularise for numerical stability (barely affects results)
    C += 1e-12 * np.eye(N_active)
    L = np.linalg.cholesky(C)

    # Draw independent standard normals: Z @ L.T ~ N(0, C)
    Z = rng.standard_normal((n_paths, N_active))
    V_active = Z @ L.T  # shape (n_paths, N_active)

    if has_zero_start:
        V_tilde = np.concatenate(
            [np.zeros((n_paths, 1)), V_active], axis=1
        )
    else:
        V_tilde = V_active

    return V_tilde, L


# ---------------------------------------------------------------------------
# Numerical conditional covariance
# ---------------------------------------------------------------------------


def volterra_conditional_covariance(
    t: float,
    s: float,
    T: float,
    H: float,
) -> float:
    """
    Compute the conditional Volterra covariance Cov(V_t^T, V_s^T) numerically.

    The covariance is given by the integral

        Cov(V_t^T, V_s^T) = integral_0^{upper} [(t - u)(s - u)]^{H - 0.5} du,

    where ``upper = min(t, s, T)`` is the largest value of u for which both
    (t - u) and (s - u) are positive (so the integrand is real-valued).  The
    integral is evaluated via ``scipy.integrate.quad``.

    Parameters
    ----------
    t : float
        First (future) time point.
    s : float
        Second (future) time point.
    T : float
        Upper integration limit (typically the VIX observation horizon or
        the current simulation time).
    H : float
        Hurst exponent, 0 < H < 0.5.

    Returns
    -------
    float
        The conditional covariance value.
    """
    upper = min(t, s, T)
    if upper <= 0.0:
        return 0.0

    def integrand(u: float) -> float:
        factor = (t - u) * (s - u)
        # Numerical guard: factor should be positive in (0, upper) by construction
        if factor <= 0.0:
            return 0.0
        return factor ** (H - 0.5)

    result, _ = quad(integrand, 0.0, upper, limit=200)
    return float(result)
