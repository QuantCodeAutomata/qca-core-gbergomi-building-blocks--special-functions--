"""
Special functions for the grey Bergomi model.

Provides three implementations of the one-parameter Mittag-Leffler function

    E_beta(z) = sum_{n=0}^{inf} z^n / Gamma(beta * n + 1),   beta in (0, 1].

For beta = 1 this reduces to exp(z).  The function is entire, so the series
converges for every z in C, but naive floating-point summation is unreliable
for large |z|; in that regime ``mittag_leffler_stable`` (backed by mpmath) is
preferred.

Public API
----------
mittag_leffler          : Truncated series in float64 arithmetic.
mittag_leffler_stable   : High-precision mpmath series via nsum.
mittag_leffler_vec      : Vectorised wrapper around mittag_leffler_stable.
build_mittag_leffler_grid : Evaluate E_beta(b * t^{2H}) over a time grid.
"""

import math
import sys
import warnings

import mpmath
import numpy as np
from scipy.special import gamma as sc_gamma

_MAX_FLOAT: float = sys.float_info.max

# ---------------------------------------------------------------------------
# Core scalar implementations
# ---------------------------------------------------------------------------


def mittag_leffler(z: float, beta: float, n_terms: int = 200) -> float:
    """
    Evaluate the one-parameter Mittag-Leffler function via a truncated series.

    E_beta(z) = sum_{n=0}^{N-1} z^n / Gamma(beta * n + 1).

    For beta = 1 the closed form exp(z) is returned directly.  For large z
    a ``RuntimeWarning`` is emitted and the result is clamped to
    ``sys.float_info.max`` to avoid silent ``inf`` propagation.

    Parameters
    ----------
    z : float
        Function argument (real).
    beta : float
        Series index, 0 < beta <= 1.
    n_terms : int, optional
        Maximum number of series terms.  The summation stops early when the
        relative contribution of the current term drops below 1e-15.
        Default is 200.

    Returns
    -------
    float
        Approximation of E_beta(z).

    Raises
    ------
    ValueError
        If beta is not in (0, 1].
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    if beta == 1.0:
        return math.exp(z)

    total: float = 0.0
    z_pow: float = 1.0  # z^n

    for n in range(n_terms):
        try:
            term = z_pow / sc_gamma(beta * n + 1.0)
        except OverflowError:
            warnings.warn(
                f"mittag_leffler: overflow at term {n} for z={z}, beta={beta}. "
                "Result clamped to sys.float_info.max.",
                RuntimeWarning,
                stacklevel=2,
            )
            return _MAX_FLOAT

        total += term

        # Early-exit: term negligible relative to running total
        if n > 0 and total != 0.0 and abs(term) <= abs(total) * 1e-15:
            break

        try:
            z_pow *= z
        except OverflowError:
            # z^n overflowed; Gamma(beta*(n+1)+1) would need to compensate —
            # at this point the next term is indeterminate, so warn and stop.
            warnings.warn(
                f"mittag_leffler: z^{n+1} overflowed for z={z}, beta={beta}. "
                "Partial sum returned; consider mittag_leffler_stable.",
                RuntimeWarning,
                stacklevel=2,
            )
            break

    return min(total, _MAX_FLOAT)


def mittag_leffler_stable(z: float, beta: float) -> float:
    """
    Evaluate E_beta(z) with high-precision mpmath arithmetic.

    Uses ``mpmath.nsum`` at 50 decimal places of working precision.  This is
    the recommended evaluator for large |z| or when accuracy matters more than
    speed.  Falls back to the truncated-series implementation if mpmath raises
    an unexpected error.

    Parameters
    ----------
    z : float
        Function argument (real).
    beta : float
        Series index, 0 < beta <= 1.

    Returns
    -------
    float
        E_beta(z), converted back to Python float.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    if beta == 1.0:
        return math.exp(z)

    try:
        mpz = mpmath.mpf(z)
        with mpmath.workdps(60):
            total = mpmath.mpf(0)
            for n in range(4000):
                term = mpz**n / mpmath.gamma(beta * n + 1.0)
                total += term
                # Convergence: relative tolerance 1e-40 after at least 10 terms
                if n >= 10 and abs(term) < mpmath.mpf("1e-40") * abs(total):
                    break
        return float(total)
    except Exception:  # pragma: no cover — safety net
        return mittag_leffler(z, beta)


# ---------------------------------------------------------------------------
# Vectorised interface
# ---------------------------------------------------------------------------


def mittag_leffler_vec(z_array: np.ndarray, beta: float) -> np.ndarray:
    """
    Vectorised Mittag-Leffler evaluation backed by mpmath.

    Applies ``mittag_leffler_stable`` element-wise to *z_array* and returns a
    numpy array of the same shape.

    Parameters
    ----------
    z_array : array-like
        Array of real arguments.
    beta : float
        Series index, 0 < beta <= 1.

    Returns
    -------
    np.ndarray
        E_beta evaluated at each element of z_array.
    """
    z = np.asarray(z_array, dtype=float)
    flat = z.ravel()
    out = np.empty_like(flat)
    for i, zi in enumerate(flat):
        out[i] = mittag_leffler_stable(float(zi), beta)
    return out.reshape(z.shape)


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------


def build_mittag_leffler_grid(
    t_grid: np.ndarray,
    b: float,
    beta: float,
    H: float,
) -> np.ndarray:
    """
    Evaluate E_beta(b * t^{2H}) for every time point in *t_grid*.

    This function is the cache-friendly entry point used by the variance-path
    simulator.  The arguments b and H encode the model's kernel scaling so
    that the Mittag-Leffler normaliser can be pre-computed once per simulation.

    For beta = 1 the fast path exp(b * t^{2H}) is used.
    For t = 0 the result is always 1 (since E_beta(0) = 1 for any beta).

    Parameters
    ----------
    t_grid : np.ndarray
        1-D array of non-negative time points.
    b : float
        Coefficient b = eta^2 * c^2 / (4 * H) from ``GBergomiParams``.
    beta : float
        Mittag-Leffler index, 0 < beta <= 1.
    H : float
        Hurst exponent, 0 < H < 0.5.

    Returns
    -------
    np.ndarray
        Array of shape ``(len(t_grid),)`` with E_beta(b * t^{2H}) for each t.
    """
    t = np.asarray(t_grid, dtype=float)
    # Compute the argument; for t=0 this naturally evaluates to 0.
    args = np.where(t > 0.0, b * t ** (2.0 * H), 0.0)

    if beta == 1.0:
        return np.exp(args)

    return mittag_leffler_vec(args, beta)
