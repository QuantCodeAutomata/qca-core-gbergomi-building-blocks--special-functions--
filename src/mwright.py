"""
M-Wright (Mainardi) random variable sampling and moment validation.

The M-Wright distribution arises naturally in the grey Bergomi model as the
distribution of the stochastic time-change Y_beta.  Its moments are

    E[Y_beta^kappa] = Gamma(1 + kappa) / Gamma(1 + beta * kappa),

and its moment-generating function is the Mittag-Leffler function E_beta.

Sampling proceeds in two steps:
  1. Draw a positive beta-stable random variable R (Laplace exponent lambda^beta).
  2. Set Y_beta = R^{-beta}.

For beta = 1, Y_1 = 1 deterministically (no randomness), recovering the
standard rough Bergomi model.

References
----------
Mainardi, Mura & Pagnini (2010) "The M-Wright Function in Time-Fractional
Diffusion Processes: A Tutorial Survey". International Journal of Differential
Equations.

Chambers, Mallows & Stuck (1976) "A Method for Simulating Stable Random
Variables". Journal of the American Statistical Association.
"""

import warnings

import numpy as np
from scipy.special import gamma as sc_gamma


# ---------------------------------------------------------------------------
# Positive stable sampler
# ---------------------------------------------------------------------------


def sample_positive_stable(
    n_samples: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample positive beta-stable random variables via Chambers-Mallows-Stuck.

    Draws R such that E[exp(-lambda * R)] = exp(-lambda^beta) for lambda >= 0.

    The algorithm uses the formula (Devroye 2009, Zolotarev parametrisation):

        R = sin(beta * U)^{1} / sin(U)^{1/beta} * (sin((1 - beta) * U) / E)^{(1-beta)/beta}

    where U ~ Uniform(0, pi) and E ~ Exponential(1).

    For beta = 1, R = 1 deterministically (degenerate stable), so an array of
    ones is returned directly.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    beta : float
        Stability index, 0 < beta <= 1.
    rng : np.random.Generator
        NumPy random generator (e.g. ``np.random.default_rng(42)``).

    Returns
    -------
    np.ndarray
        Shape ``(n_samples,)`` array of positive stable variates.

    Raises
    ------
    ValueError
        If beta is not in (0, 1].
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    if beta == 1.0:
        return np.ones(n_samples)

    # U ~ Uniform(0, pi); draw strictly inside (0, pi) to avoid sin(0)=0.
    U = rng.uniform(low=0.0, high=np.pi, size=n_samples)
    E = rng.exponential(scale=1.0, size=n_samples)

    # Guard against the (measure-zero) event U = 0 or U = pi.
    eps = np.finfo(float).tiny
    U = np.clip(U, eps, np.pi - eps)

    sin_U = np.sin(U)
    sin_betaU = np.sin(beta * U)
    sin_1mbetaU = np.sin((1.0 - beta) * U)
    exponent = (1.0 - beta) / beta

    R = (sin_betaU / sin_U ** (1.0 / beta)) * (sin_1mbetaU / E) ** exponent

    if np.any(R <= 0):
        n_bad = int(np.sum(R <= 0))
        warnings.warn(
            f"sample_positive_stable: {n_bad} non-positive values encountered; "
            "replaced with machine epsilon.",
            RuntimeWarning,
            stacklevel=2,
        )
        R = np.where(R > 0, R, np.finfo(float).tiny)

    return R


# ---------------------------------------------------------------------------
# M-Wright sampler
# ---------------------------------------------------------------------------


def sample_mwright(
    n_samples: int,
    beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Sample M-Wright (Mainardi) random variables Y_beta = R^{-beta}.

    The M-Wright distribution with index beta ∈ (0, 1) is the law of R^{-beta}
    where R is a positive beta-stable random variable with Laplace transform
    exp(-lambda^beta).  Its moments satisfy

        E[Y_beta^kappa] = Gamma(1 + kappa) / Gamma(1 + beta * kappa).

    For beta = 1, Y_1 = 1 deterministically, so an array of ones is returned.

    Parameters
    ----------
    n_samples : int
        Number of samples to draw.
    beta : float
        M-Wright index, 0 < beta <= 1.
    rng : np.random.Generator
        NumPy random generator.

    Returns
    -------
    np.ndarray
        Shape ``(n_samples,)`` array of M-Wright variates.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    if beta == 1.0:
        return np.ones(n_samples)

    R = sample_positive_stable(n_samples, beta, rng)
    return R ** (-beta)


# ---------------------------------------------------------------------------
# Moment helpers
# ---------------------------------------------------------------------------


def theoretical_moment(kappa: float, beta: float) -> float:
    """
    Return the kappa-th moment of Y_beta under the M-Wright distribution.

    E[Y_beta^kappa] = Gamma(1 + kappa) / Gamma(1 + beta * kappa).

    Parameters
    ----------
    kappa : float
        Moment order (must be > -1 so that Gamma(1 + kappa) is finite).
    beta : float
        M-Wright index, 0 < beta <= 1.

    Returns
    -------
    float
        Theoretical moment value.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    return sc_gamma(1.0 + kappa) / sc_gamma(1.0 + beta * kappa)


def validate_moments(
    beta: float,
    kappas: list[float],
    n_samples: int = 100_000,
    seed: int = 42,
) -> dict:
    """
    Compare empirical moments of sampled Y_beta to the theoretical formula.

    Draws *n_samples* M-Wright variates and computes E[Y_beta^kappa] for each
    kappa in *kappas*, then contrasts with Gamma(1+kappa)/Gamma(1+beta*kappa).

    Parameters
    ----------
    beta : float
        M-Wright index, 0 < beta <= 1.
    kappas : list of float
        Moment orders to validate (e.g. [0.5, 1.0, 1.5, 2.0]).
    n_samples : int, optional
        Monte Carlo sample size.  Default is 100 000.
    seed : int, optional
        Random seed for reproducibility.  Default is 42.

    Returns
    -------
    dict
        Dictionary with keys:

        ``'kappa'``
            List of moment orders (mirrors the input *kappas*).
        ``'empirical'``
            List of empirical moment estimates.
        ``'theoretical'``
            List of exact theoretical moments.
        ``'abs_error'``
            List of absolute errors |empirical - theoretical|.
        ``'rel_error'``
            List of relative errors |empirical - theoretical| / theoretical.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    rng = np.random.default_rng(seed)
    samples = sample_mwright(n_samples, beta, rng)

    empirical: list[float] = []
    theoretical: list[float] = []
    abs_err: list[float] = []
    rel_err: list[float] = []

    for kappa in kappas:
        emp = float(np.mean(samples**kappa))
        thy = theoretical_moment(kappa, beta)
        ae = abs(emp - thy)
        re = ae / abs(thy) if thy != 0.0 else float("nan")

        empirical.append(emp)
        theoretical.append(thy)
        abs_err.append(ae)
        rel_err.append(re)

    return {
        "kappa": list(kappas),
        "empirical": empirical,
        "theoretical": theoretical,
        "abs_error": abs_err,
        "rel_error": rel_err,
    }
