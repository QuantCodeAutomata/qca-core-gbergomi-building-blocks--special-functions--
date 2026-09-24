"""
Black-Scholes implied volatility utilities.

Provides scalar and vectorised implied-volatility inversion for European call
options under zero interest rates and continuous dividends, as used throughout
the gBergomi experiments.

Uses py_vollib_vectorized when available for speed; falls back to a scipy
brentq root-finding loop otherwise.

Custom — Context7 found no single library that covers all required IV-inversion
conventions (zero-rate, normalized payoff, robust handling of deep OTM/ITM).
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm


# ---------------------------------------------------------------------------
# Black-Scholes helpers (zero rate)
# ---------------------------------------------------------------------------


def bs_call_price(
    S: float | np.ndarray,
    K: float | np.ndarray,
    T: float,
    sigma: float | np.ndarray,
) -> float | np.ndarray:
    """
    Black-Scholes European call price under zero interest rate.

    Parameters
    ----------
    S : float or array
        Spot price (forward = spot under zero rate).
    K : float or array
        Strike price.
    T : float
        Time to maturity in years.
    sigma : float or array
        Implied volatility.

    Returns
    -------
    float or np.ndarray
        Call price(s).
    """
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return S * norm.cdf(d1) - K * norm.cdf(d2)


def bs_implied_vol_scalar(
    price: float,
    S: float,
    K: float,
    T: float,
    sigma_lo: float = 1e-6,
    sigma_hi: float = 10.0,
    tol: float = 1e-8,
) -> float:
    """
    Scalar Black-Scholes implied volatility via Brent's root-finding.

    Parameters
    ----------
    price : float
        Observed call price.
    S : float
        Spot price.
    K : float
        Strike price.
    T : float
        Time to maturity.
    sigma_lo, sigma_hi : float, optional
        Bracket for Brent's method.
    tol : float, optional
        Convergence tolerance.

    Returns
    -------
    float
        Implied volatility, or NaN if inversion fails.
    """
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + tol or T <= 0:
        return float("nan")

    try:
        iv = brentq(
            lambda sig: bs_call_price(S, K, T, sig) - price,
            sigma_lo,
            sigma_hi,
            xtol=tol,
            maxiter=200,
        )
        return iv
    except (ValueError, RuntimeError):
        return float("nan")


def implied_vol_vectorized(
    prices: np.ndarray,
    S: float,
    K_grid: np.ndarray,
    T: float,
) -> np.ndarray:
    """
    Vectorised implied volatility inversion over a strike grid.

    Attempts to use py_vollib_vectorized for speed; falls back to a scalar loop
    via brentq.

    Parameters
    ----------
    prices : np.ndarray
        Call prices, shape (n_K,).
    S : float
        Spot price.
    K_grid : np.ndarray
        Strike grid, shape (n_K,).
    T : float
        Time to maturity.

    Returns
    -------
    np.ndarray
        Implied volatilities, shape (n_K,).  NaN where inversion failed.
    """
    try:
        from py_vollib_vectorized import vectorized_implied_volatility as viv

        flags = np.full(len(K_grid), "c")
        iv = viv(
            price=prices,
            S=np.full(len(K_grid), S),
            K=K_grid,
            t=np.full(len(K_grid), T),
            r=np.zeros(len(K_grid)),
            flag=flags,
            q=np.zeros(len(K_grid)),
            return_as="numpy",
            on_error="nan",
        )
        return np.asarray(iv, dtype=float)

    except Exception:
        # Scalar fallback
        ivs = np.empty(len(K_grid))
        for j, K in enumerate(K_grid):
            ivs[j] = bs_implied_vol_scalar(prices[j], S, K, T)
        return ivs


def atm_smile_fit(
    log_moneyness: np.ndarray,
    iv: np.ndarray,
    degree: int = 2,
) -> tuple[float, float, float]:
    """
    Fit a quadratic IV(k) = a + b*k + c*k^2 and return level, skew, curvature.

    Only non-NaN IV values are used for the fit.

    Parameters
    ----------
    log_moneyness : np.ndarray
        Log-moneyness k = log(K/S).
    iv : np.ndarray
        Implied volatility.
    degree : int, optional
        Polynomial degree (default 2 for quadratic).

    Returns
    -------
    tuple[float, float, float]
        (level=a, skew=b, curvature=c) from the quadratic fit.
        Returns (NaN, NaN, NaN) if fit fails.
    """
    mask = np.isfinite(iv)
    if mask.sum() < degree + 1:
        return float("nan"), float("nan"), float("nan")
    try:
        coeffs = np.polyfit(log_moneyness[mask], iv[mask], degree)
        # polyfit returns coefficients in descending order
        if degree == 2:
            c, b, a = coeffs
        elif degree == 1:
            b, a = coeffs
            c = 0.0
        else:
            a = coeffs[-1]
            b = coeffs[-2] if len(coeffs) > 1 else 0.0
            c = coeffs[-3] if len(coeffs) > 2 else 0.0
        return float(a), float(b), float(c)
    except (np.linalg.LinAlgError, ValueError):
        return float("nan"), float("nan"), float("nan")
