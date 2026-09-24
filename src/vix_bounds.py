"""
VIX futures analytic bounds under the gBergomi model.

Implements the lower and upper bounds for VIX futures stated in Proposition 2.15
of "Rough Bergomi Turns Grey":

    Upper bound: UB(T) = sqrt( (1/Delta) * integral_T^{T+Delta} xi_0(s) ds )

    Lower bound: LB(T) = sqrt( (1/Delta) * integral_T^{T+Delta}
                               xi_0(s) / E_beta(b * s^{2H}) ds )

The lower bound exploits the convexity adjustment due to M-Wright randomization,
while the upper bound follows from Jensen's inequality applied to the square-root.
Note: E_beta(b*s^{2H}) >= 1 for positive arguments, so LB <= UB.

Custom implementation — Context7 found no library covering gBergomi VIX bounds.
"""

from collections.abc import Callable

import numpy as np
from scipy.integrate import quad

from src.config import GBergomiParams
from src.special_functions import mittag_leffler_stable


def _integrand_upper(s: float, xi0_func: Callable[[float], float]) -> float:
    """Integrand for the upper bound: xi_0(s)."""
    return xi0_func(s)


def _integrand_lower(
    s: float,
    xi0_func: Callable[[float], float],
    b: float,
    beta: float,
    H: float,
) -> float:
    """Integrand for the lower bound: xi_0(s) / E_beta(b * s^{2H})."""
    eb = mittag_leffler_stable(b * s ** (2.0 * H), beta)
    return xi0_func(s) / eb


def compute_vix_bounds(
    params: GBergomiParams,
    xi0_func: Callable[[float], float],
    T_grid: np.ndarray,
    Delta: float = 1.0 / 12.0,
    n_quad_points: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Proposition 2.15 lower and upper bounds for VIX futures.

    For each maturity T the bounds are:

        UB(T) = sqrt( (1/Delta) * integral_T^{T+Delta} xi_0(s) ds )

        LB(T) = sqrt( (1/Delta) * integral_T^{T+Delta} xi_0(s)/E_beta(b s^{2H}) ds )

    The lower bound is always <= the upper bound because E_beta(z) >= 1 for z > 0.

    Parameters
    ----------
    params : GBergomiParams
        Model parameters; uses params.b, params.beta, params.H.
    xi0_func : callable
        Forward-variance curve xi_0(t) -> float.
    T_grid : np.ndarray
        1-D array of maturities T, shape (M,).
    Delta : float, optional
        VIX integration window (one month convention). Default 1/12.
    n_quad_points : int, optional
        Limit argument passed to scipy.integrate.quad. Default 50.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (lb, ub) — arrays of shape (M,) with lower and upper bounds.
    """
    b, beta, H = params.b, params.beta, params.H
    n_T = len(T_grid)
    lb = np.empty(n_T)
    ub = np.empty(n_T)

    for i, T in enumerate(T_grid):
        lo, hi = T, T + Delta

        # Upper bound integrand: xi_0(s)
        ub_integral, _ = quad(_integrand_upper, lo, hi, args=(xi0_func,), limit=n_quad_points)
        ub[i] = np.sqrt(ub_integral / Delta)

        # Lower bound integrand: xi_0(s) / E_beta(b * s^{2H})
        lb_integral, _ = quad(
            _integrand_lower, lo, hi, args=(xi0_func, b, beta, H), limit=n_quad_points
        )
        lb[i] = np.sqrt(lb_integral / Delta)

    # Sanity check: LB <= UB up to floating-point noise
    violation = np.sum(lb > ub + 1e-8)
    if violation > 0:
        import warnings
        warnings.warn(
            f"compute_vix_bounds: {violation} points where LB > UB; "
            "possible numerical issue in Mittag-Leffler evaluation.",
            RuntimeWarning,
            stacklevel=2,
        )

    return lb, ub
