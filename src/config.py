"""
Configuration: model parameters and forward-variance curve scenarios for the grey
Bergomi (gBergomi) model.

The gBergomi model generalises the rough Bergomi model of Bayer–Friz–Gatheral by
replacing the Gaussian driver with an M-Wright (Mainardi) random variable indexed by
beta ∈ (0, 1].  Setting beta = 1 recovers the classical rough Bergomi model.

References
----------
Guennoun, Jacquier, Roome & Shi (2018) "Asymptotic behaviour of the fractional
Heston model". SIAM Journal on Financial Mathematics.
"""

import math
from dataclasses import dataclass

from scipy.special import gamma as sc_gamma


# ---------------------------------------------------------------------------
# Model parameter dataclass
# ---------------------------------------------------------------------------

@dataclass
class GBergomiParams:
    """
    Parameters for the grey Bergomi (gBergomi) model.

    Parameters
    ----------
    H : float
        Hurst roughness exponent.  Must satisfy 0 < H < 0.5 for a rough
        volatility process.
    beta : float
        M-Wright / Mittag-Leffler index.  Must satisfy 0 < beta <= 1.
        When beta = 1 the model reduces to the standard rough Bergomi model.
    eta : float
        Vol-of-vol parameter (positive).
    rho : float, optional
        Correlation between the spot Brownian motion and the Volterra driver.
        Default is 0.0 (no leverage).

    Derived properties (read-only)
    --------------------------------
    H_plus  : H + 0.5   (the kernel exponent H - 1/2 shifted up by 1)
    H_minus : H - 0.5   (negative for rough paths)
    c       : 1 / Gamma(H + 0.5)
    b       : eta^2 * c^2 / (4 * H)  — coefficient in E_beta(b * t^{2H})
    """

    H: float
    beta: float
    eta: float
    rho: float = 0.0

    def __post_init__(self) -> None:
        if not (0.0 < self.H < 0.5):
            raise ValueError(f"H must be in (0, 0.5), got {self.H}")
        if not (0.0 < self.beta <= 1.0):
            raise ValueError(f"beta must be in (0, 1], got {self.beta}")
        if self.eta <= 0.0:
            raise ValueError(f"eta must be positive, got {self.eta}")
        if not (-1.0 <= self.rho <= 1.0):
            raise ValueError(f"rho must be in [-1, 1], got {self.rho}")

    # ------------------------------------------------------------------
    # Derived properties
    # ------------------------------------------------------------------

    @property
    def H_plus(self) -> float:
        """H + 0.5 — the Volterra kernel exponent offset."""
        return self.H + 0.5

    @property
    def H_minus(self) -> float:
        """H - 0.5 — negative for rough-volatility regimes (H < 0.5)."""
        return self.H - 0.5

    @property
    def c(self) -> float:
        """Normalisation constant 1 / Gamma(H + 0.5)."""
        return 1.0 / sc_gamma(self.H + 0.5)

    @property
    def b(self) -> float:
        """Coefficient b = eta^2 * c^2 / (4 * H) for E_beta(b * t^{2H})."""
        return self.eta**2 * self.c**2 / (4.0 * self.H)

    def __repr__(self) -> str:
        return (
            f"GBergomiParams(H={self.H}, beta={self.beta}, eta={self.eta}, "
            f"rho={self.rho}, c={self.c:.6f}, b={self.b:.6f})"
        )


# ---------------------------------------------------------------------------
# Forward-variance curve scenarios
# ---------------------------------------------------------------------------

_XI_BASE: float = 0.235**2  # ≈ 0.055225


def xi_scenario1(t: float) -> float:
    """
    Scenario 1: flat forward-variance curve.

    xi_0(t) = 0.235^2  for all t >= 0.

    Parameters
    ----------
    t : float
        Time in years.

    Returns
    -------
    float
        Constant forward variance 0.235^2.
    """
    return _XI_BASE


def xi_scenario2(t: float) -> float:
    """
    Scenario 2: quadratically increasing forward-variance curve.

    xi_0(t) = 0.235^2 * (1 + t)^2.

    Parameters
    ----------
    t : float
        Time in years.

    Returns
    -------
    float
        Forward variance at time t.
    """
    return _XI_BASE * (1.0 + t) ** 2


def xi_scenario3(t: float) -> float:
    """
    Scenario 3: slowly increasing forward-variance curve.

    xi_0(t) = 0.235^2 * sqrt(1 + t).

    Parameters
    ----------
    t : float
        Time in years.

    Returns
    -------
    float
        Forward variance at time t.
    """
    return _XI_BASE * math.sqrt(1.0 + t)


# ---------------------------------------------------------------------------
# Pre-defined paper parameter sets
# ---------------------------------------------------------------------------

PAPER_PARAMS: dict[str, GBergomiParams] = {
    "fig1": GBergomiParams(H=0.065, beta=0.037, eta=0.65),
    "spx_base": GBergomiParams(H=0.065, beta=1.0, eta=1.09, rho=-1.0),
}
"""
Named parameter sets taken from the gBergomi literature.

``fig1``
    Parameters used for the VIX-smile figure (beta << 1, M-Wright regime).
``spx_base``
    SPX-calibrated base case with full leverage (rho = -1).
"""

# ---------------------------------------------------------------------------
# Model-wide constants
# ---------------------------------------------------------------------------

DELTA: float = 1.0 / 12.0
"""VIX averaging window: one calendar month (1/12 of a year)."""

SEED: int = 42
"""Default random seed for reproducible Monte Carlo experiments."""
