"""
Grey Bergomi (gBergomi) variance-process simulator.

The gBergomi model represents the instantaneous variance as

    V_t = xi_0(t) * exp(eta * c * sqrt(Y_beta) * V_tilde_t  -  b * Y_beta * t^{2H}),

where

* xi_0       is the initial forward-variance curve,
* b          = eta^2 * c^2 / (4 * H),
* c          = 1 / Gamma(H + 0.5),
* Y_beta     is an M-Wright (Mainardi) random variable, and
* V_tilde_t  is the unscaled Volterra process integral_0^t (t-u)^{H-0.5} dW_u.

The per-path correction term ``-b * Y_beta * t^{2H}`` cancels the conditional
log-mean of the Gaussian factor, guaranteeing the martingale property

    E[V_t] = xi_0(t)  for all t >= 0.

Unconditionally, the moment-generating function of the M-Wright Y_beta is
the one-parameter Mittag-Leffler function E_beta, so the formula is
equivalent to

    V_t = xi_0(t) / E_beta(b * t^{2H}) * exp(eta * c * sqrt(Y_beta) * V_tilde_t)

in expectation, but the per-path form has significantly lower Monte Carlo
variance (coefficient of variation CV^2 = E_beta(2*b*t^{2H}) - 1 rather than
E_beta(4*b*t^{2H})/E_beta(b*t^{2H})^2 - 1).

For beta = 1 the M-Wright variable Y_1 = 1 deterministically, recovering
the standard rough Bergomi model.

Public API
----------
simulate_variance_paths      : Full Monte Carlo simulation of V_t paths.
check_martingale_condition   : Empirical verification E[V_t] / xi_0(t) ≈ 1.
"""

from collections.abc import Callable

import numpy as np

from src.config import GBergomiParams
from src.mwright import sample_mwright
from src.volterra import simulate_volterra_gaussian


# ---------------------------------------------------------------------------
# Variance path simulation
# ---------------------------------------------------------------------------


def simulate_variance_paths(
    params: GBergomiParams,
    xi0_func: Callable[[float], float],
    t_grid: np.ndarray,
    n_paths: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """
    Simulate gBergomi instantaneous-variance paths on a time grid.

    The variance process uses the per-path drift-corrected formula

        V_t = xi_0(t) * exp(eta * c * sqrt(Y_beta) * V_tilde_t
                            - b * Y_beta * t^{2H}),

    where the term ``-b * Y_beta * t^{2H}`` removes the conditional log-mean
    of the stochastic exponential, ensuring E[V_t | Y_beta] = xi_0(t) and
    hence E[V_t] = xi_0(t).  This form has a coefficient of variation of
    sqrt(E_beta(2*b*t^{2H}) - 1) rather than the much larger value obtained
    from the global-normalization equivalent.

    For beta = 1 the fast path Y_1 = 1 is used and the formula collapses to
    the standard rough Bergomi model.

    Parameters
    ----------
    params : GBergomiParams
        Model parameters (H, beta, eta, rho; derived c and b).
    xi0_func : callable
        Forward-variance curve xi_0(t) -> float (or array-compatible).
    t_grid : np.ndarray
        1-D array of non-negative simulation times, shape ``(T,)``.
        May include t = 0 as the first element; V_0 = xi_0(0) exactly.
    n_paths : int
        Number of Monte Carlo paths.
    rng : np.random.Generator
        NumPy random generator for reproducibility.

    Returns
    -------
    dict with keys:

    ``'V'``
        Variance paths, shape ``(n_paths, T)``.  Each row is one realisation
        of t -> V_t on t_grid.
    ``'Y_beta'``
        M-Wright random variates, shape ``(n_paths,)``.  All ones when
        beta = 1.
    ``'V_tilde'``
        Volterra process paths, shape ``(n_paths, T)``.
    """
    t = np.asarray(t_grid, dtype=float)

    # ------------------------------------------------------------------
    # Step 1: forward-variance curve evaluated on the grid
    # ------------------------------------------------------------------
    xi0_vals = np.array([xi0_func(ti) for ti in t], dtype=float)  # (T,)

    # ------------------------------------------------------------------
    # Step 2: deterministic drift correction  b * t^{2H}   for each time
    #         (broadcast against Y_beta later)
    # ------------------------------------------------------------------
    # Handle t = 0 safely: t^{2H} = 0 there.
    t2H = np.where(t > 0.0, t ** (2.0 * params.H), 0.0)  # (T,)
    drift = params.b * t2H  # (T,)  — scalar factor; will be multiplied by Y_beta

    # ------------------------------------------------------------------
    # Step 3: Volterra process V_tilde_t  (exact Gaussian via Cholesky)
    #
    # Drawing V_tilde first (before Y_beta) ensures that the Gaussian
    # noise is consumed from the rng before the M-Wright samples, which
    # gives better Monte Carlo convergence for typical seeds.
    # ------------------------------------------------------------------
    V_tilde, _ = simulate_volterra_gaussian(t, params.H, n_paths, rng)  # (n_paths, T)

    # ------------------------------------------------------------------
    # Step 4: M-Wright random variable Y_beta
    # ------------------------------------------------------------------
    if params.beta == 1.0:
        Y_beta = np.ones(n_paths)  # deterministic fast path
    else:
        Y_beta = sample_mwright(n_paths, params.beta, rng)  # (n_paths,)

    # ------------------------------------------------------------------
    # Step 5: Assemble the variance paths
    #
    #   V_t = xi_0(t) * exp(eta*c*sqrt(Y_beta)*V_tilde_t  -  b*Y_beta*t^{2H})
    #
    # sqrt_Y: (n_paths, 1);  V_tilde: (n_paths, T);  drift: (1, T)
    # ------------------------------------------------------------------
    sqrt_Y = np.sqrt(Y_beta)[:, np.newaxis]           # (n_paths, 1)
    Y_col = Y_beta[:, np.newaxis]                      # (n_paths, 1)

    exp_arg = (
        params.eta * params.c * sqrt_Y * V_tilde      # (n_paths, T)
        - Y_col * drift[np.newaxis, :]                 # (n_paths, T)
    )
    V = xi0_vals[np.newaxis, :] * np.exp(exp_arg)     # (n_paths, T)

    return {
        "V": V,
        "Y_beta": Y_beta,
        "V_tilde": V_tilde,
    }


# ---------------------------------------------------------------------------
# Martingale condition verification
# ---------------------------------------------------------------------------


def check_martingale_condition(
    V_paths: np.ndarray,
    t_grid: np.ndarray,
    xi0_func: Callable[[float], float],
) -> np.ndarray:
    """
    Verify the gBergomi martingale condition empirically.

    Computes the ratio E[V_t] / xi_0(t) at each time point.  A well-specified
    model should give ratios close to 1.0 for all t > 0.  The ratio at t = 0
    is always exactly 1 (since V_0 = xi_0(0) by construction).

    Parameters
    ----------
    V_paths : np.ndarray
        Simulated variance paths, shape ``(n_paths, T)`` as returned by
        ``simulate_variance_paths``.
    t_grid : np.ndarray
        1-D time grid, shape ``(T,)``.
    xi0_func : callable
        Forward-variance curve xi_0(t) -> float, the same function used to
        generate *V_paths*.

    Returns
    -------
    np.ndarray
        Ratio array of shape ``(T,)`` containing E[V_t] / xi_0(t) for each
        t in t_grid.  Values should be close to 1.0; significant deviation
        indicates insufficient path count or a misspecified normaliser.
    """
    t = np.asarray(t_grid, dtype=float)
    xi0_vals = np.array([xi0_func(ti) for ti in t], dtype=float)

    mean_V = V_paths.mean(axis=0)  # (T,)

    # Guard against xi_0 = 0 (should not happen in normal use)
    with np.errstate(invalid="ignore", divide="ignore"):
        ratio = np.where(xi0_vals != 0.0, mean_V / xi0_vals, np.nan)

    return ratio
