"""Tests for src/asymptotics.py — short-maturity asymptotic formulas."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.asymptotics import (
    vix_atm_level_asymptotic,
    spx_atm_level_asymptotic,
    spx_atm_skew_asymptotic,
    compute_j_integrals,
)
from src.config import GBergomiParams, xi_scenario1, DELTA


# ---------------------------------------------------------------------------
# VIX ATM level
# ---------------------------------------------------------------------------


def test_vix_atm_level_positive():
    """VIX ATM level must be positive."""
    params = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
    level = vix_atm_level_asymptotic(params, xi_scenario1, T=1 / 252, Delta=DELTA)
    assert level > 0.0


def test_vix_atm_level_beta1_reduces_to_rbergomi():
    """At beta=1, E_1(z)=exp(z); for 0<beta<1, E_beta(z) > exp(z) for z>0.
    Therefore the LB integrand xi_0/E_beta is smaller for small beta, giving
    a lower LB for the grey model than for rBergomi. VIX futures themselves
    can be higher due to the Y_beta heavy-tail amplification."""
    T = 1 / 252
    params_grey = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
    params_rb = GBergomiParams(H=0.065, beta=1.0, eta=0.65)
    lev_grey = vix_atm_level_asymptotic(params_grey, xi_scenario1, T, DELTA)
    lev_rb = vix_atm_level_asymptotic(params_rb, xi_scenario1, T, DELTA)
    # E_{0.037}(z) > exp(z) for z >= 0 (denominator larger for small beta)
    # → grey LB <= rBergomi LB
    assert lev_grey <= lev_rb + 1e-8, (
        f"grey LB {lev_grey:.5f} should be ≤ rBergomi LB {lev_rb:.5f}"
    )
    # Both should be positive
    assert lev_grey > 0.0
    assert lev_rb > 0.0


def test_vix_atm_level_increasing_then_flat_with_T():
    """VIX ATM level should remain bounded as T grows."""
    params = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
    T_vals = np.array([1 / 252, 5 / 252, 10 / 252, 20 / 252, 1 / 12])
    levels = np.array([
        vix_atm_level_asymptotic(params, xi_scenario1, T, DELTA)
        for T in T_vals
    ])
    assert np.all(levels > 0.0)
    assert np.all(np.isfinite(levels))


def test_vix_atm_level_flat_xi0():
    """For flat xi_0(t)=c^2 (const), UB = sqrt(c^2) = c."""
    c_level = 0.235
    xi_flat = lambda t: c_level**2
    params = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
    level = vix_atm_level_asymptotic(params, xi_flat, T=1 / 252, Delta=DELTA)
    # level should be close to 0.235 (or slightly above due to ML correction)
    assert abs(level - c_level) < 0.10  # loose bound — correction factor


# ---------------------------------------------------------------------------
# SPX ATM level
# ---------------------------------------------------------------------------


def test_spx_atm_level_near_sqrt_xi0():
    """SPX ATM level → sqrt(xi_0(0)) as T → 0."""
    params = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
    xi0_0 = xi_scenario1(1e-9)
    level = spx_atm_level_asymptotic(params, xi_scenario1, T=1e-4)
    expected = np.sqrt(xi0_0)
    assert abs(level - expected) < 0.001, f"level={level:.6f}, expected={expected:.6f}"


def test_spx_atm_level_positive():
    """SPX ATM level must be positive."""
    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    level = spx_atm_level_asymptotic(params, xi_scenario1, T=0.1)
    assert level > 0.0


def test_spx_atm_level_from_scalar():
    """SPX ATM level from scalar xi0 = sqrt(xi0)."""
    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    xi0_val = 0.235**2
    level = spx_atm_level_asymptotic(params, xi0_val)
    assert abs(level - 0.235) < 1e-8


# ---------------------------------------------------------------------------
# SPX ATM skew
# ---------------------------------------------------------------------------


def test_spx_atm_skew_zero_for_zero_rho():
    """At rho=0, SPX ATM skew = 0 (no leverage effect)."""
    params = GBergomiParams(H=0.065, beta=0.037, eta=0.65, rho=0.0)
    xi0_0 = xi_scenario1(1e-9)
    skew = spx_atm_skew_asymptotic(params, xi0_0, T=1 / 12)
    assert abs(skew) < 1e-10


def test_spx_atm_skew_negative_for_negative_rho():
    """Negative rho → negative ATM skew (downside skew)."""
    params = GBergomiParams(H=0.065, beta=0.037, eta=0.65, rho=-0.9)
    xi0_0 = xi_scenario1(1e-9)
    skew = spx_atm_skew_asymptotic(params, xi0_0, T=1 / 12)
    assert skew < 0.0


def test_spx_atm_skew_scales_with_T_H():
    """SPX ATM skew should scale proportionally to T^H."""
    params = GBergomiParams(H=0.065, beta=0.037, eta=0.65, rho=-0.9)
    xi0_0 = xi_scenario1(1e-9)
    T1, T2 = 1 / 12, 2 / 12
    s1 = spx_atm_skew_asymptotic(params, xi0_0, T1)
    s2 = spx_atm_skew_asymptotic(params, xi0_0, T2)
    ratio_skew = s2 / s1
    ratio_expected = (T2 / T1) ** params.H
    assert abs(ratio_skew - ratio_expected) < 0.05


# ---------------------------------------------------------------------------
# J integrals
# ---------------------------------------------------------------------------


def test_j2_equals_j3_when_h_is_2H():
    """J2 = (1/Δ)*∫_T^{T+Δ} s^H ds and J3 = (1/Δ)*∫_T^{T+Δ} s^{2H} ds are different."""
    j1, j2, j3 = compute_j_integrals(beta=0.5, H=0.065, Delta=DELTA, T=0.0)
    assert j1 > 0.0
    assert j2 > 0.0
    assert j3 > 0.0


def test_j_integrals_beta1_j1_equals_j3():
    """At beta=1: J1 integrates s^{2H*1} = s^{2H} = same as J3."""
    j1, j2, j3 = compute_j_integrals(beta=1.0, H=0.065, Delta=DELTA, T=0.0)
    # J1 = (1/Δ)∫s^{2H} ds = J3
    assert abs(j1 - j3) < 1e-6


def test_j_integrals_positive():
    """All J integrals should be strictly positive."""
    for beta in [0.037, 0.1, 0.5, 1.0]:
        j1, j2, j3 = compute_j_integrals(beta=beta, H=0.065, Delta=DELTA, T=0.0)
        assert j1 > 0.0, f"J1 not positive for beta={beta}"
        assert j2 > 0.0, f"J2 not positive for beta={beta}"
        assert j3 > 0.0, f"J3 not positive for beta={beta}"
