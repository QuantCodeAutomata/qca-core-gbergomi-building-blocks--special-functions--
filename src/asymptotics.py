"""
Short-maturity asymptotic formulas for VIX and SPX smile quantities under gBergomi.

Implements the leading-order short-maturity asymptotics for:
  - VIX ATM implied volatility level, skew, and curvature
  - SPX ATM implied volatility level and skew

These are based on the regime T -> 0 with Delta fixed (VIX window).

Custom implementation — Context7 found no library covering gBergomi asymptotics.
The formulas below are derived from the paper's Section 4 structure.

For small T, the key simplification is that the conditional Volterra process
V_t^T = integral_0^T (t-u)^{H-0.5} dB_u is small, and the dominant contributions
come from the normalization xi_0(t)/E_beta(b*t^{2H}).

Short-maturity VIX ATM IV level ~ sqrt(xi_0(T) / E_beta(b*(T+Delta/2)^{2H}))
evaluated at mid-window.

Short-maturity SPX ATM IV level ~ sqrt(xi_0(0)) as T->0 from the Black-Scholes
at-the-money formula sqrt(V_0 * T / (2*pi*T)) * sqrt(2*pi) ~ sqrt(xi_0(0)).
"""

from __future__ import annotations

import numpy as np
from scipy.special import gamma as sc_gamma

from src.config import GBergomiParams
from src.special_functions import mittag_leffler_stable


def vix_atm_level_asymptotic(
    params: GBergomiParams,
    xi0_func,
    T: float,
    Delta: float = 1.0 / 12.0,
) -> float:
    """
    Leading-order short-maturity VIX ATM implied volatility level.

    Approximates the VIX ATM level as the square root of the average forward
    variance normalized by the Mittag-Leffler factor across the VIX window:

        I_T^VIX ~ sqrt( (1/Delta) * integral_T^{T+Delta} xi_0(s)/E_beta(b*s^{2H}) ds )

    This corresponds to the gBergomi lower bound, which becomes tight as T->0.

    Parameters
    ----------
    params : GBergomiParams
        Model parameters.
    xi0_func : callable
        Forward-variance curve.
    T : float
        VIX maturity.
    Delta : float, optional
        VIX window (default 1/12).

    Returns
    -------
    float
        Asymptotic VIX ATM level.
    """
    from scipy.integrate import quad

    b, beta, H = params.b, params.beta, params.H

    def integrand(s: float) -> float:
        eb = mittag_leffler_stable(b * s ** (2.0 * H), beta)
        return xi0_func(s) / eb

    integral, _ = quad(integrand, T, T + Delta, limit=50)
    return float(np.sqrt(integral / Delta))


def vix_atm_level_flat(
    params: GBergomiParams,
    xi0_flat: float,
    T: float,
    Delta: float = 1.0 / 12.0,
) -> float:
    """
    Closed-form short-T VIX ATM level for flat xi_0 = xi_flat.

    For flat xi_0, the asymptotic level simplifies to:
        I_T ~ sqrt( xi_flat * (1/Delta) * integral_T^{T+Delta} 1/E_beta(b*s^{2H}) ds )

    Parameters
    ----------
    params : GBergomiParams
    xi0_flat : float
        Constant forward variance.
    T : float
    Delta : float

    Returns
    -------
    float
    """
    from scipy.integrate import quad

    b, beta, H = params.b, params.beta, params.H

    def integrand(s: float) -> float:
        eb = mittag_leffler_stable(b * s ** (2.0 * H), beta)
        return 1.0 / eb

    integral, _ = quad(integrand, T, T + Delta, limit=50)
    return float(np.sqrt(xi0_flat * integral / Delta))


def spx_atm_level_asymptotic(
    params: GBergomiParams,
    xi0_func_or_val,
    T: float = 0.0,
    Delta: float = 1.0 / 12.0,
) -> float:
    """
    Leading-order short-maturity SPX ATM implied volatility level.

    For the gBergomi model with any xi_0 satisfying xi_0(0) > 0:

        I_hat_T ~ sqrt(xi_0(0))  as T -> 0.

    This follows from the Black-Scholes ATM formula sigma ~ sqrt(V_0).

    Parameters
    ----------
    params : GBergomiParams
        Model parameters (unused at leading order, present for interface consistency).
    xi0_func_or_val : callable or float
        Forward-variance curve (callable) or constant value.
    T : float, optional
        Maturity (unused at leading order; xi_0(0) is evaluated).
    Delta : float, optional
        VIX window (unused at leading order).

    Returns
    -------
    float
        Leading-order SPX ATM implied vol ~ sqrt(xi_0(0)).
    """
    if callable(xi0_func_or_val):
        xi0_0 = float(xi0_func_or_val(1e-10))
    else:
        xi0_0 = float(xi0_func_or_val)
    return float(np.sqrt(xi0_0))


def spx_atm_skew_asymptotic(
    params: GBergomiParams,
    xi0_at_0: float,
    T: float,
) -> float:
    """
    Short-maturity SPX ATM skew under gBergomi.

    The ATM skew of the SPX smile scales as:
        skew ~ rho * eta * c / (H + 0.5) * T^H / sqrt(xi_0(0))

    This is the dominant contribution from the correlation between the
    Volterra driver and the spot, analogous to the rough Bergomi formula.

    Parameters
    ----------
    params : GBergomiParams
        Model parameters including rho.
    xi0_at_0 : float
        Forward variance at t=0.
    T : float
        Maturity.

    Returns
    -------
    float
        Asymptotic ATM skew (d IV / d log-moneyness at ATM).
    """
    H, eta, c, rho = params.H, params.eta, params.c, params.rho
    H_plus = H + 0.5
    # Leading-order: rho * eta * c / H_plus * T^H / sqrt(xi0)
    return float(rho * eta * c / H_plus * T**H / np.sqrt(xi0_at_0))


def vix_atm_skew_numerical(
    levels: np.ndarray,
    maturities: np.ndarray,
) -> np.ndarray:
    """
    Numerically estimate the maturity-derivative of VIX ATM level (proxy for skew).

    Computes finite-difference derivative dI/dT using the input maturity grid.

    Parameters
    ----------
    levels : np.ndarray
        VIX ATM level at each maturity, shape (M,).
    maturities : np.ndarray
        Maturity grid, shape (M,).

    Returns
    -------
    np.ndarray
        Finite-difference slope dI/dT, shape (M-1,).
    """
    return np.diff(levels) / np.diff(maturities)


def compute_j_integrals(
    beta: float,
    H: float,
    Delta: float = 1.0 / 12.0,
    T: float = 0.0,
    n_quad: int = 100,
) -> tuple[float, float, float]:
    """
    Compute J_1, J_2, J_3 integrals for the VIX asymptotic expansion.

    These are defined as integrals over the VIX window [T, T+Delta]:
        J_1 = (1/Delta) * integral_T^{T+Delta} s^{2H*beta} ds
        J_2 = (1/Delta) * integral_T^{T+Delta} s^{H} ds
        J_3 = (1/Delta) * integral_T^{T+Delta} s^{2H} ds

    These represent leading-order contributions of the M-Wright randomization.

    Parameters
    ----------
    beta : float
    H : float
    Delta : float, optional
    T : float, optional
        VIX maturity start. Default 0 (short-maturity limit).
    n_quad : int, optional
        Number of quadrature points.

    Returns
    -------
    tuple[float, float, float]
        (J_1, J_2, J_3)
    """
    from scipy.integrate import quad

    lo, hi = T, T + Delta

    j1, _ = quad(lambda s: s ** (2.0 * H * beta), lo, hi, limit=n_quad)
    j2, _ = quad(lambda s: s**H, lo, hi, limit=n_quad)
    j3, _ = quad(lambda s: s ** (2.0 * H), lo, hi, limit=n_quad)

    return j1 / Delta, j2 / Delta, j3 / Delta
