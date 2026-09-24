"""
Integration tests for experiment modules.

Tests that experiment functions run without errors and produce
valid outputs (correct types, shapes, and value ranges).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest

from src.config import GBergomiParams, xi_scenario1, DELTA, SEED
from src.special_functions import mittag_leffler_stable
from src.mwright import sample_mwright
from src.gbergomi import simulate_variance_paths
from src.vix_bounds import compute_vix_bounds
from src.implied_vol import bs_call_price, implied_vol_vectorized
from src.asymptotics import (
    vix_atm_level_asymptotic, spx_atm_level_asymptotic,
    spx_atm_skew_asymptotic, compute_j_integrals
)


# ---------------------------------------------------------------------------
# VIX Monte Carlo (Exp 3)
# ---------------------------------------------------------------------------

def test_vix_mc_output_positive():
    """VIX paths from Monte Carlo should be strictly positive."""
    from exp.exp3_vix_mc import simulate_vix_paths

    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    rng = np.random.default_rng(SEED)
    vix = simulate_vix_paths(params, xi_scenario1, T=0.25, Delta=DELTA,
                              N_tau=10, n_paths=200, rng=rng, l=4)
    assert vix.shape == (200,)
    assert np.all(vix > 0.0)


def test_vix_mc_beta1_close_to_xi0():
    """Beta=1 (rough Bergomi): E[VIX_T] near sqrt(xi_0) for small T."""
    from exp.exp3_vix_mc import simulate_vix_paths

    params = GBergomiParams(H=0.065, beta=1.0, eta=0.65)
    rng = np.random.default_rng(SEED)
    vix = simulate_vix_paths(params, xi_scenario1, T=0.05, Delta=DELTA,
                              N_tau=10, n_paths=500, rng=rng, l=4)
    mean_vix = float(np.mean(vix))
    # E[VIX_T] ≈ sqrt(xi_0(0)) at small T
    expected = np.sqrt(xi_scenario1(0.1))
    assert abs(mean_vix - expected) < 0.05, f"mean VIX={mean_vix:.4f}, expected≈{expected:.4f}"


def test_vix_mc_mean_between_bounds():
    """VIX MC mean should lie between Prop 2.15 lower and upper bounds."""
    from exp.exp3_vix_mc import simulate_vix_paths

    T = 0.5
    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    rng = np.random.default_rng(SEED)
    vix = simulate_vix_paths(params, xi_scenario1, T=T, Delta=DELTA,
                              N_tau=10, n_paths=300, rng=rng, l=4)
    mean_vix = float(np.mean(vix))

    lb, ub = compute_vix_bounds(params, xi_scenario1, np.array([T]), DELTA)
    # Allow some Monte Carlo noise
    assert mean_vix >= lb[0] - 0.03, f"VIX mean={mean_vix:.4f} below LB={lb[0]:.4f}"
    assert mean_vix <= ub[0] + 0.05, f"VIX mean={mean_vix:.4f} above UB={ub[0]:.4f}"


# ---------------------------------------------------------------------------
# SPX Monte Carlo (Exp 4)
# ---------------------------------------------------------------------------

def test_spx_mc_martingale():
    """SPX martingale: E[S_T] ≈ S_0 = 1 under zero rates."""
    from exp.exp4_spx_mc import simulate_spx_paths

    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65, rho=-0.5)
    rng = np.random.default_rng(SEED)
    n_steps = 20
    log_ST = simulate_spx_paths(params, xi_scenario1, T=0.5, n_steps=n_steps,
                                  n_paths=2_000, rng=rng)
    E_ST = float(np.mean(np.exp(log_ST)))
    assert abs(E_ST - 1.0) < 0.05, f"E[S_T]={E_ST:.4f} should be near 1"


def test_spx_mc_negative_rho_skew():
    """Negative rho → negatively skewed log-return distribution."""
    from exp.exp4_spx_mc import simulate_spx_paths
    from scipy import stats

    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65, rho=-0.9)
    rng = np.random.default_rng(SEED)
    log_ST = simulate_spx_paths(params, xi_scenario1, T=0.25, n_steps=15,
                                  n_paths=2_000, rng=rng)
    skew = float(stats.skew(log_ST))
    assert skew < 0, f"Expected negative skew, got {skew:.4f}"


def test_spx_mc_price_call_non_negative():
    """Call prices must be non-negative."""
    from exp.exp4_spx_mc import simulate_spx_paths, price_spx_calls

    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65, rho=-0.5)
    rng = np.random.default_rng(SEED)
    log_ST = simulate_spx_paths(params, xi_scenario1, T=0.25, n_steps=15,
                                  n_paths=500, rng=rng)
    K_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    prices = price_spx_calls(log_ST, K_grid, S0=1.0)
    assert np.all(prices >= 0.0)
    assert np.all(np.isfinite(prices))


def test_spx_mc_call_monotone_in_K():
    """Call prices should be monotone decreasing in K."""
    from exp.exp4_spx_mc import simulate_spx_paths, price_spx_calls

    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65, rho=-0.5)
    rng = np.random.default_rng(SEED)
    log_ST = simulate_spx_paths(params, xi_scenario1, T=0.25, n_steps=15,
                                  n_paths=1_000, rng=rng)
    K_grid = np.array([0.8, 0.9, 1.0, 1.1, 1.2])
    prices = price_spx_calls(log_ST, K_grid, S0=1.0)
    assert np.all(np.diff(prices) <= 0.01), "Call prices should be decreasing in K"


# ---------------------------------------------------------------------------
# Empirical VIX analysis (Exp 8) — offline unit tests only
# ---------------------------------------------------------------------------

def test_log_returns_computation():
    """Log returns computed correctly from price series."""
    from exp.exp8_vix_empirical import compute_log_returns
    import pandas as pd

    prices = pd.Series([20.0, 21.0, 19.5, 22.0], index=pd.date_range("2020-01-01", periods=4))
    returns = compute_log_returns(prices)
    expected = np.log(np.array([21.0, 19.5, 22.0]) / np.array([20.0, 21.0, 19.5]))
    np.testing.assert_allclose(returns.values, expected, rtol=1e-10)


def test_run_normality_tests_output():
    """Normality tests return dict with correct keys and valid values."""
    from exp.exp8_vix_empirical import run_normality_tests
    import pandas as pd

    rng = np.random.default_rng(0)
    returns = pd.Series(rng.standard_normal(500))
    results = run_normality_tests(returns)

    required_keys = [
        "n", "mean", "std", "skewness", "excess_kurtosis",
        "dagostino_skew_stat", "dagostino_skew_pval",
        "anscombe_kurt_stat", "anscombe_kurt_pval",
        "omnibus_stat", "omnibus_pval",
        "adf_stat", "adf_pval",
    ]
    for k in required_keys:
        assert k in results, f"Missing key: {k}"
    assert results["n"] == 500
    assert 0.0 <= results["adf_pval"] <= 1.0


def test_gaussian_returns_fail_kurtosis_test_less():
    """Gaussian returns should NOT reject kurtosis test at 1% level (most of the time)."""
    from exp.exp8_vix_empirical import run_normality_tests
    import pandas as pd

    # Large Gaussian sample: kurtosis test should not reject at 1% (with high probability)
    rng = np.random.default_rng(42)
    returns = pd.Series(rng.standard_normal(2000))
    results = run_normality_tests(returns)
    # excess kurtosis should be close to 0
    assert abs(results["excess_kurtosis"]) < 0.5


def test_heavy_tailed_rejects_normality():
    """Cauchy-distributed returns should strongly reject normality."""
    from exp.exp8_vix_empirical import run_normality_tests
    import pandas as pd

    rng = np.random.default_rng(123)
    returns = pd.Series(rng.standard_cauchy(500))
    results = run_normality_tests(returns)
    # Cauchy has very high kurtosis → should reject normality
    assert results["omnibus_pval"] < 0.05


def test_get_subperiod():
    """Subperiod slicing returns correct date range."""
    from exp.exp8_vix_empirical import get_subperiod, compute_log_returns
    import pandas as pd

    dates = pd.date_range("2005-01-01", "2022-12-31", freq="B")
    prices = pd.Series(np.ones(len(dates)) * 20.0, index=dates)
    returns = compute_log_returns(prices)

    sub = get_subperiod(returns, "2010-01-01", "2014-12-31")
    assert all(sub.index >= pd.Timestamp("2010-01-01"))
    assert all(sub.index <= pd.Timestamp("2014-12-31"))
