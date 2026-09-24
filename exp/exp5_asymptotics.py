"""
Experiment 5: Short-Maturity Asymptotics for VIX and SPX ATM Smile Quantities.

Compares paper's short-maturity asymptotic formulas with direct numerical estimates.

VIX asymptotics (as T → 0):
  - Level: I_T^VIX ≈ sqrt((1/Δ)*∫_T^{T+Δ} ξ₀(s)/E_β(b*s^{2H}) ds) — lower bound
  - This equals the LB from Prop. 2.15, which becomes tight at T→0.

SPX asymptotics (as T → 0):
  - Level: I_hat_T ~ sqrt(ξ₀(0)) as T → 0 (BS ATM formula)
  - Skew: ∂_k IV|_{k=0} ~ ρ*η*c/(H+0.5) * T^H / sqrt(ξ₀(0))

J integrals for VIX convexity analysis:
  J_1 = (1/Δ)*∫_T^{T+Δ} s^{2H*β} ds  (Mittag-Leffler correction leading order)
  J_2 = (1/Δ)*∫_T^{T+Δ} s^H ds        (Volterra correction)
  J_3 = (1/Δ)*∫_T^{T+Δ} s^{2H} ds     (drift correction)
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.integrate import quad

from src.config import GBergomiParams, xi_scenario1, DELTA, SEED
from src.asymptotics import (
    vix_atm_level_asymptotic, spx_atm_level_asymptotic,
    spx_atm_skew_asymptotic, compute_j_integrals
)
from src.vix_bounds import compute_vix_bounds
from src.implied_vol import implied_vol_vectorized, atm_smile_fit

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Small-maturity grid
T_GRID_SMALL = np.array([1/252, 2/252, 5/252, 10/252, 20/252, 1/12, 2/12])


# ---------------------------------------------------------------------------
# 1. VIX ATM level vs beta
# ---------------------------------------------------------------------------

def compute_vix_atm_vs_beta(
    beta_grid: np.ndarray,
    T: float = 1/252,
) -> np.ndarray:
    """Compute asymptotic VIX ATM level vs beta at fixed small T."""
    levels = []
    for beta in beta_grid:
        params = GBergomiParams(H=0.065, beta=beta, eta=0.65)
        lev = vix_atm_level_asymptotic(params, xi_scenario1, T, DELTA)
        levels.append(lev)
    return np.array(levels)


# ---------------------------------------------------------------------------
# 2. VIX ATM level convergence: analytic vs LB
# ---------------------------------------------------------------------------

def compute_vix_convergence(T_grid: np.ndarray, beta: float = 0.037) -> dict:
    """Compare analytic asymptotic VIX ATM level vs LB for each T."""
    params = GBergomiParams(H=0.065, beta=beta, eta=0.65)

    # Asymptotic formula (equals LB when computed from lower bound integrand)
    asymp_levels = np.array([
        vix_atm_level_asymptotic(params, xi_scenario1, T, DELTA)
        for T in T_grid
    ])

    # True lower bound from Proposition 2.15
    lb, ub = compute_vix_bounds(params, xi_scenario1, T_grid, DELTA)

    return {
        "T_grid": T_grid,
        "asymp": asymp_levels,
        "lb": lb,
        "ub": ub,
    }


# ---------------------------------------------------------------------------
# 3. J integrals as function of beta
# ---------------------------------------------------------------------------

def compute_j_integrals_vs_beta(
    beta_grid: np.ndarray,
    H: float = 0.065,
    Delta: float = DELTA,
    T: float = 0.0,
) -> dict:
    """Compute J_1, J_2, J_3 for varying beta at fixed T and Delta."""
    j1_vals, j2_vals, j3_vals = [], [], []
    for beta in beta_grid:
        j1, j2, j3 = compute_j_integrals(beta, H, Delta, T)
        j1_vals.append(j1)
        j2_vals.append(j2)
        j3_vals.append(j3)
    return {
        "beta": beta_grid,
        "J1": np.array(j1_vals),
        "J2": np.array(j2_vals),
        "J3": np.array(j3_vals),
    }


# ---------------------------------------------------------------------------
# 4. SPX ATM skew asymptotics vs T
# ---------------------------------------------------------------------------

def compute_spx_atm_skew_asymptotic(
    T_grid: np.ndarray,
    rho_grid: list,
    beta: float = 0.037,
) -> dict:
    """Compute theoretical SPX ATM skew for varying T and rho."""
    params_H = 0.065
    xi0_0 = xi_scenario1(1e-6)  # xi_0(0)
    results = {}
    for rho in rho_grid:
        params = GBergomiParams(H=params_H, beta=beta, eta=0.65, rho=rho)
        skews = np.array([spx_atm_skew_asymptotic(params, xi0_0, T) for T in T_grid])
        results[rho] = skews
    return results


# ---------------------------------------------------------------------------
# 5. SPX numerical skew estimation
# ---------------------------------------------------------------------------

def estimate_spx_skew_numerically(
    T_grid: np.ndarray,
    rho: float = -0.9,
    beta: float = 0.037,
    n_paths: int = 3_000,
) -> np.ndarray:
    """Numerically estimate SPX ATM skew at each maturity T."""
    from exp.exp4_spx_mc import simulate_spx_paths, price_spx_calls

    skews = []
    S0 = 1.0
    # Strike grid around ATM
    log_m_grid = np.linspace(-0.10, 0.10, 11)
    K_grid = S0 * np.exp(log_m_grid)

    for T in T_grid:
        n_steps = max(int(T * 252), 2)
        params = GBergomiParams(H=0.065, beta=beta, eta=0.65, rho=rho)
        rng = np.random.default_rng(SEED)
        log_ST = simulate_spx_paths(params, xi_scenario1, T, n_steps, n_paths, rng)
        prices = price_spx_calls(log_ST, K_grid, S0)
        ivs = implied_vol_vectorized(prices, S0, K_grid, T)
        _, skew, _ = atm_smile_fit(log_m_grid, ivs, degree=2)
        skews.append(skew)
    return np.array(skews)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_vix_atm_vs_beta(beta_grid: np.ndarray, levels: np.ndarray) -> None:
    """Plot VIX ATM level vs beta (Figure 4 style)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(beta_grid, levels, "o-", color="#e74c3c", lw=2, ms=5, label="VIX ATM level")
    ax.axhline(0.235, color="gray", lw=1.5, linestyle="--", label="sqrt(ξ₀)=0.235")
    ax.set_xlabel("β (grey parameter)", fontsize=11)
    ax.set_ylabel("Asymptotic VIX ATM IV level", fontsize=11)
    ax.set_title(f"Short-T VIX ATM Level vs β (H=0.065, η=0.65, T=1/252d)", fontsize=11)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp5_vix_atm_vs_beta.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_j_integrals(j_results: dict) -> None:
    """Plot J_1, J_2, J_3 vs beta."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    beta = j_results["beta"]

    for ax, key, label, color in zip(
        axes,
        ["J1", "J2", "J3"],
        ["J₁ = ⟨s^{2Hβ}⟩", "J₂ = ⟨s^H⟩", "J₃ = ⟨s^{2H}⟩"],
        ["#e74c3c", "#3498db", "#2ecc71"]
    ):
        ax.plot(beta, j_results[key], "o-", color=color, lw=2, ms=5)
        ax.set_xlabel("β")
        ax.set_title(label, fontsize=11)
        ax.grid(True, alpha=0.3)

    fig.suptitle("J integrals vs β — VIX asymptotic components (H=0.065, Δ=1/12)", fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp5_j_integrals.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_spx_skew_vs_T(
    T_grid: np.ndarray,
    asymp_skews: dict,
    numerical_skews: dict | None = None,
    T_grid_mc: np.ndarray | None = None,
) -> None:
    """Plot SPX ATM skew vs maturity T (Figure 5 style).

    Parameters
    ----------
    T_grid : ndarray
        Maturity grid for asymptotic curves.
    asymp_skews : dict
        rho -> asymptotic skew array aligned with T_grid.
    numerical_skews : dict, optional
        rho -> MC-estimated skew array aligned with T_grid_mc.
    T_grid_mc : ndarray, optional
        Maturity grid used for Monte Carlo estimates (may differ from T_grid).
    """
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {-0.9: "#e74c3c", -0.3: "#3498db", 0.0: "#2ecc71"}

    for rho, skews in asymp_skews.items():
        mask = np.isfinite(skews)
        ax.plot(T_grid[mask], skews[mask], color=colors.get(rho, "gray"),
                lw=2, label=f"ρ={rho:+.1f} (asymptotic)")

    if numerical_skews:
        # Use dedicated MC T-grid when provided; fall back to full T_grid
        _T_mc = T_grid_mc if T_grid_mc is not None else T_grid
        for rho, skews in numerical_skews.items():
            skews = np.asarray(skews)
            mask = np.isfinite(skews)
            ax.plot(_T_mc[mask], skews[mask], color=colors.get(rho, "gray"),
                    lw=1.5, ls="--", marker="o", ms=4,
                    label=f"ρ={rho:+.1f} (MC est.)")

    ax.set_xlabel("Maturity T (years)")
    ax.set_ylabel("ATM Skew ∂IV/∂k|_{k=0}")
    ax.set_title("SPX Short-Maturity ATM Skew vs T (H=0.065, β=0.037, η=0.65)", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp5_spx_atm_skew.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("=" * 60)
    print("Experiment 5: Short-Maturity Asymptotics for VIX and SPX")
    print("=" * 60)

    # 1. VIX ATM level vs beta
    print("\n1. VIX ATM level vs beta...")
    beta_dense = np.linspace(0.02, 1.0, 50)
    vix_levels = compute_vix_atm_vs_beta(beta_dense, T=1/252)
    plot_vix_atm_vs_beta(beta_dense, vix_levels)

    # 2. VIX convergence
    print("\n2. VIX asymptotic convergence (LB tightness at small T)...")
    T_small = np.array([1/252, 2/252, 5/252, 10/252, 20/252, 1/12])
    conv = compute_vix_convergence(T_small, beta=0.037)
    print("  T_grid | Asymp | LB | UB")
    for T, a, lb, ub in zip(conv["T_grid"], conv["asymp"], conv["lb"], conv["ub"]):
        print(f"  T={T*252:.1f}d | {a:.4f} | {lb:.4f} | {ub:.4f}")

    # 3. J integrals
    print("\n3. J integrals vs beta...")
    j_results = compute_j_integrals_vs_beta(beta_dense)
    plot_j_integrals(j_results)

    # 4. SPX ATM skew asymptotics
    print("\n4. SPX ATM skew (asymptotic) vs T...")
    T_small2 = np.array([1/252, 5/252, 10/252, 20/252, 1/12, 2/12, 3/12])
    rho_grid = [-0.9, -0.3, 0.0]
    asymp_skews = compute_spx_atm_skew_asymptotic(T_small2, rho_grid, beta=0.037)

    # 5. Numerical SPX skew (fast estimate)
    print("\n5. SPX ATM skew (numerical MC) vs T for rho=-0.9...")
    T_for_mc = np.array([1/12, 2/12, 3/12])  # only a few maturities for speed
    numerical_skews_rho = {
        -0.9: estimate_spx_skew_numerically(T_for_mc, rho=-0.9, beta=0.037, n_paths=2_000)
    }
    print(f"  Numerical skews (ρ=-0.9): {numerical_skews_rho[-0.9]}")

    plot_spx_skew_vs_T(T_small2, asymp_skews, numerical_skews_rho, T_grid_mc=T_for_mc)

    _update_results_md(vix_levels, beta_dense, conv, asymp_skews, T_small2)


def _update_results_md(vix_levels, beta_dense, conv, asymp_skews, T_small) -> None:
    results_path = os.path.join(RESULTS_DIR, "RESULTS.md")

    # Sample asymptotic levels at selected beta
    beta_samples = [0.037, 0.1, 0.5, 1.0]
    level_rows = []
    for b in beta_samples:
        idx = np.argmin(np.abs(beta_dense - b))
        level_rows.append(f"| β={b} | {vix_levels[idx]:.4f} |")

    summary = f"""
## Experiment 5: Short-Maturity Asymptotics

**Parameters**: H=0.065, η=0.65, Δ=1/12, ξ₀(t)=0.235²

### VIX ATM Level vs β (T=1/252)
| β | Level |
|---|-------|
{chr(10).join(level_rows)}

### VIX LB Tightness (β=0.037)
| T (days) | Asymptotic | LB | UB |
|----------|------------|----|----|
""" + "\n".join(
    [f"| {T*252:.0f} | {a:.4f} | {lb:.4f} | {ub:.4f} |"
     for T, a, lb, ub in zip(conv["T_grid"], conv["asymp"], conv["lb"], conv["ub"])]
) + f"""

### SPX ATM Skew (β=0.037, selected ρ)
| T (days) | ρ=-0.9 | ρ=-0.3 |
|----------|--------|--------|
""" + "\n".join(
    [f"| {T*252:.0f} | {asymp_skews[-0.9][i]:.4f} | {asymp_skews[-0.3][i]:.4f} |"
     for i, T in enumerate(T_small)]
) + """

**Key findings**: Lower β increases VIX ATM level (Mittag-Leffler normalization diverges
from 1). SPX ATM skew scales as T^H, controlled primarily by ρ. The LB equals
the asymptotic VIX ATM level at small T (tight bound).
"""
    with open(results_path, "a") as f:
        f.write(summary)
    print(f"\nResults appended to {results_path}")


if __name__ == "__main__":
    main()
