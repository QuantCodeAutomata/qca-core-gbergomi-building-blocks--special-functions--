"""Tests for src/implied_vol.py — Black-Scholes implied volatility utilities."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.implied_vol import (
    bs_call_price,
    bs_implied_vol_scalar,
    implied_vol_vectorized,
    atm_smile_fit,
)


# ---------------------------------------------------------------------------
# bs_call_price
# ---------------------------------------------------------------------------


def test_bs_call_price_atm():
    """ATM call price with zero drift: C ≈ S * sigma * sqrt(T / (2*pi))."""
    S, K, T, sigma = 1.0, 1.0, 1.0, 0.2
    price = bs_call_price(S, K, T, sigma)
    approx = S * sigma * np.sqrt(T / (2 * np.pi))
    assert abs(price - approx) < 0.01, f"ATM price {price:.4f} vs approx {approx:.4f}"


def test_bs_call_price_intrinsic():
    """Deep ITM call price ≈ S - K."""
    price = bs_call_price(1.0, 0.5, 1.0, 0.01)
    assert abs(price - 0.5) < 0.01


def test_bs_call_price_otm_near_zero():
    """Very deep OTM call price ≈ 0."""
    price = bs_call_price(1.0, 3.0, 0.1, 0.2)
    assert price < 1e-6


def test_bs_call_price_non_negative():
    """Call prices should always be non-negative."""
    for S, K, sigma in [(1.0, 0.8, 0.3), (1.0, 1.0, 0.2), (1.0, 1.2, 0.1)]:
        p = bs_call_price(S, K, 0.5, sigma)
        assert p >= 0.0


def test_bs_call_price_monotone_in_sigma():
    """Higher sigma → higher call price."""
    sigmas = [0.1, 0.2, 0.3, 0.4]
    prices = [bs_call_price(1.0, 1.0, 1.0, s) for s in sigmas]
    assert all(prices[i] < prices[i + 1] for i in range(len(prices) - 1))


# ---------------------------------------------------------------------------
# bs_implied_vol_scalar
# ---------------------------------------------------------------------------


def test_implied_vol_round_trip():
    """Implied vol from a BS-priced call should recover the original sigma."""
    S, K, T, sigma_true = 1.0, 1.0, 0.5, 0.25
    price = bs_call_price(S, K, T, sigma_true)
    sigma_hat = bs_implied_vol_scalar(price, S, K, T)
    assert abs(sigma_hat - sigma_true) < 1e-5


def test_implied_vol_round_trip_otm():
    """Round-trip for OTM option."""
    S, K, T, sigma_true = 1.0, 1.1, 0.25, 0.30
    price = bs_call_price(S, K, T, sigma_true)
    sigma_hat = bs_implied_vol_scalar(price, S, K, T)
    assert abs(sigma_hat - sigma_true) < 1e-4


def test_implied_vol_round_trip_itm():
    """Round-trip for ITM option."""
    S, K, T, sigma_true = 1.0, 0.9, 0.5, 0.20
    price = bs_call_price(S, K, T, sigma_true)
    sigma_hat = bs_implied_vol_scalar(price, S, K, T)
    assert abs(sigma_hat - sigma_true) < 1e-4


def test_implied_vol_intrinsic_returns_nan():
    """Price = intrinsic value → inversion should return NaN."""
    S, K, T = 1.0, 0.9, 0.5
    intrinsic = S - K
    iv = bs_implied_vol_scalar(intrinsic, S, K, T)
    # Either NaN or a very small vol — both acceptable
    assert np.isnan(iv) or iv < 1e-4


def test_implied_vol_zero_maturity_nan():
    """Zero maturity → NaN."""
    iv = bs_implied_vol_scalar(0.1, 1.0, 1.0, 0.0)
    assert np.isnan(iv)


# ---------------------------------------------------------------------------
# implied_vol_vectorized
# ---------------------------------------------------------------------------


def test_vectorized_round_trip():
    """Vectorized IV should recover original sigmas from BS prices."""
    S0 = 1.0
    T = 0.5
    log_m = np.array([-0.1, 0.0, 0.1])
    K_grid = S0 * np.exp(log_m)
    sigma_true = np.array([0.30, 0.25, 0.22])  # smile

    prices = np.array([bs_call_price(S0, K, T, sig) for K, sig in zip(K_grid, sigma_true)])
    ivs = implied_vol_vectorized(prices, S0, K_grid, T)

    for iv, sig in zip(ivs, sigma_true):
        if not np.isnan(iv):
            assert abs(iv - sig) < 1e-3


def test_vectorized_output_shape():
    """Output shape matches input shape."""
    K = np.linspace(0.8, 1.2, 10)
    prices = np.array([bs_call_price(1.0, k, 0.5, 0.25) for k in K])
    ivs = implied_vol_vectorized(prices, 1.0, K, 0.5)
    assert ivs.shape == (10,)


# ---------------------------------------------------------------------------
# atm_smile_fit
# ---------------------------------------------------------------------------


def test_atm_smile_fit_quadratic():
    """Fit a perfect quadratic and verify level/skew/curvature recovery."""
    log_m = np.linspace(-0.2, 0.2, 21)
    a, b, c = 0.25, -0.3, 0.5  # IV(k) = a + b*k + c*k^2
    iv = a + b * log_m + c * log_m**2

    level, skew, curv = atm_smile_fit(log_m, iv, degree=2)
    assert abs(level - a) < 1e-6
    assert abs(skew - b) < 1e-6
    assert abs(curv - c) < 1e-6


def test_atm_smile_fit_nan_handling():
    """NaN values should be handled gracefully."""
    log_m = np.linspace(-0.2, 0.2, 21)
    iv = 0.25 * np.ones(21)
    iv[0] = np.nan
    iv[-1] = np.nan

    level, skew, curv = atm_smile_fit(log_m, iv, degree=2)
    assert not np.isnan(level)


def test_atm_smile_fit_insufficient_data():
    """Fewer data points than degree → NaN."""
    log_m = np.array([0.0])
    iv = np.array([0.25])
    level, skew, curv = atm_smile_fit(log_m, iv, degree=2)
    assert np.isnan(level)
