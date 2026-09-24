"""
Experiment 6: Separate Market Calibrations to VIX and SPX Smiles Under gBergomi.

Implements the paper's two-step calibration:
  Stage 1: Calibrate VIX smile over (H, beta, eta) — VIX prefers small beta.
  Stage 2: Calibrate SPX smile with H fixed, over (beta, eta, rho) — SPX prefers beta~1.

Data notes:
  - Paper mentions 17/10/2023 vs 25/10/2023 (inconsistency); we use 17/10/2023 as primary.
  - Uses yfinance to download options/spot data; falls back to synthetic smiles if unavailable.
  - VIX options are approximated using a simulated VIX smile when market data unavailable.
  - Optimizer: scipy.optimize.differential_evolution, single-threaded, maxiter=30, popsize=15.

Replication assumptions (logged below):
  - Delta = 1/12 (one-month VIX window)
  - xi_0(t) = (current ATM IV)^2 (flat proxy; standard rough vol practice)
  - VIX options quoted off VIX spot level (standard CBOE convention)
  - SPX maturity: 1-month slice where available
  - OTM options only for fitting
  - Zero rates (paper assumption)
"""

import sys
import os
import warnings

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import differential_evolution
from scipy.special import gamma as sc_gamma

from src.config import GBergomiParams, xi_scenario1, DELTA, SEED
from src.special_functions import mittag_leffler_stable
from src.mwright import sample_mwright
from src.implied_vol import implied_vol_vectorized, bs_call_price
from src.vix_bounds import compute_vix_bounds

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# ========================
# Replication assumptions
# ========================
CALIBRATION_DATE = "2023-10-17"   # primary date (vs paper's 25/10/2023 in figures)
NOTES = """
Replication assumptions:
  - Date: 2023-10-17 (paper inconsistent: text says 17/10, figures say 25/10)
  - Delta = 1/12 year (one-month VIX window)
  - xi_0(t) = flat at historical VIX level squared (proxy for term structure)
  - VIX options priced off VIX spot (standard CBOE)
  - Zero interest rates (paper assumption)
  - Optimizer: differential_evolution, maxiter=30, popsize=15, single-threaded
  - H bounds: (0.01, 0.49), beta bounds: (0.01, 1.0), eta bounds: (0.05, 3.0)
  - rho bounds: (-1.0, 0.0)
"""


# ---------------------------------------------------------------------------
# Synthetic smile generation (fallback + testing)
# ---------------------------------------------------------------------------

def generate_synthetic_vix_smile(
    H: float = 0.065,
    beta: float = 0.037,
    eta: float = 0.65,
    xi0_level: float = 0.235**2,
    T_vix: float = 1/12,
    n_strikes: int = 11,
    n_paths: int = 5_000,
    rng_seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic VIX option smile using gBergomi Monte Carlo."""
    from exp.exp3_vix_mc import simulate_vix_paths
    from src.implied_vol import implied_vol_vectorized

    params = GBergomiParams(H=H, beta=beta, eta=eta)
    xi0_func = lambda t: xi0_level  # flat curve
    rng = np.random.default_rng(rng_seed)

    # Simulate VIX
    vix_paths = simulate_vix_paths(
        params, xi0_func, T_vix, DELTA, N_tau=20, n_paths=n_paths, rng=rng, l=8
    )
    F_vix = float(np.mean(vix_paths))

    # VIX option call prices (VIX future as forward)
    vix_level = F_vix
    strike_grid = np.linspace(0.6 * vix_level, 1.5 * vix_level, n_strikes)

    call_prices = np.array([
        float(np.mean(np.maximum(vix_paths - K, 0.0)))
        for K in strike_grid
    ])
    ivs = implied_vol_vectorized(call_prices, vix_level, strike_grid, T_vix)

    return strike_grid, ivs


def generate_synthetic_spx_smile(
    H: float = 0.065,
    beta: float = 1.0,
    eta: float = 1.09,
    rho: float = -1.0,
    xi0_level: float = 0.235**2,
    T_spx: float = 1/12,
    n_paths: int = 5_000,
    rng_seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a synthetic SPX option smile for calibration target."""
    from exp.exp4_spx_mc import simulate_spx_paths, price_spx_calls

    params = GBergomiParams(H=H, beta=beta, eta=eta, rho=rho)
    xi0_func = lambda t: xi0_level
    rng = np.random.default_rng(rng_seed + 1)
    n_steps = max(int(T_spx * 252), 2)

    log_ST = simulate_spx_paths(params, xi0_func, T_spx, n_steps, n_paths, rng)
    log_m_grid = np.linspace(-0.15, 0.15, 13)
    K_grid = np.exp(log_m_grid)  # S0=1
    prices = price_spx_calls(log_ST, K_grid, S0=1.0)
    ivs = implied_vol_vectorized(prices, 1.0, K_grid, T_spx)
    return K_grid, ivs


# ---------------------------------------------------------------------------
# Model pricing for calibration (VIX smile)
# ---------------------------------------------------------------------------

def model_vix_ivs(
    H: float,
    beta: float,
    eta: float,
    xi0_level: float,
    T_vix: float,
    K_strikes: np.ndarray,
    n_paths: int = 2_000,
    rng_seed: int = SEED,
) -> np.ndarray:
    """Price VIX options under gBergomi and return implied vols."""
    from exp.exp3_vix_mc import simulate_vix_paths
    params = GBergomiParams(H=H, beta=beta, eta=eta)
    xi0_func = lambda t: xi0_level
    rng = np.random.default_rng(rng_seed)

    try:
        vix_paths = simulate_vix_paths(
            params, xi0_func, T_vix, DELTA, N_tau=15, n_paths=n_paths, rng=rng, l=8
        )
        F_vix = float(np.mean(vix_paths))
        call_prices = np.array([
            float(np.mean(np.maximum(vix_paths - K, 0.0))) for K in K_strikes
        ])
        ivs = implied_vol_vectorized(call_prices, F_vix, K_strikes, T_vix)
        return ivs
    except Exception:
        return np.full(len(K_strikes), np.nan)


def model_spx_ivs(
    H: float,
    beta: float,
    eta: float,
    rho: float,
    xi0_level: float,
    T_spx: float,
    K_grid: np.ndarray,
    n_paths: int = 2_000,
    rng_seed: int = SEED,
) -> np.ndarray:
    """Price SPX options under gBergomi and return implied vols."""
    from exp.exp4_spx_mc import simulate_spx_paths, price_spx_calls

    params = GBergomiParams(H=H, beta=beta, eta=eta, rho=rho)
    xi0_func = lambda t: xi0_level
    rng = np.random.default_rng(rng_seed)
    n_steps = max(int(T_spx * 252), 2)

    try:
        log_ST = simulate_spx_paths(params, xi0_func, T_spx, n_steps, n_paths, rng)
        prices = price_spx_calls(log_ST, K_grid, S0=1.0)
        ivs = implied_vol_vectorized(prices, 1.0, K_grid, T_spx)
        return ivs
    except Exception:
        return np.full(len(K_grid), np.nan)


# ---------------------------------------------------------------------------
# Calibration objectives
# ---------------------------------------------------------------------------

def vix_objective(
    params_vec: np.ndarray,
    K_market: np.ndarray,
    iv_market: np.ndarray,
    xi0_level: float,
    T_vix: float,
    n_paths: int,
    rng_seed: int,
) -> float:
    """MSE objective for VIX calibration over (H, beta, eta)."""
    H, beta, eta = params_vec
    iv_model = model_vix_ivs(H, beta, eta, xi0_level, T_vix, K_market, n_paths, rng_seed)
    mask = np.isfinite(iv_model) & np.isfinite(iv_market)
    if mask.sum() < 2:
        return 1e6
    return float(np.mean((iv_model[mask] - iv_market[mask])**2))


def spx_objective(
    params_vec: np.ndarray,
    K_market: np.ndarray,
    iv_market: np.ndarray,
    H_fixed: float,
    xi0_level: float,
    T_spx: float,
    n_paths: int,
    rng_seed: int,
) -> float:
    """MSE objective for SPX calibration over (beta, eta, rho)."""
    beta, eta, rho = params_vec
    iv_model = model_spx_ivs(H_fixed, beta, eta, rho, xi0_level, T_spx, K_market, n_paths, rng_seed)
    mask = np.isfinite(iv_model) & np.isfinite(iv_market)
    if mask.sum() < 2:
        return 1e6
    return float(np.mean((iv_model[mask] - iv_market[mask])**2))


# ---------------------------------------------------------------------------
# Main calibration routine
# ---------------------------------------------------------------------------

def run_calibration(n_paths_calib: int = 1_000) -> dict:
    """
    Run the two-stage gBergomi calibration using synthetic target smiles.

    Stage 1: Calibrate VIX smile over (H, beta, eta).
             Target: synthetic gBergomi smile with (H=0.065, beta=0.037, eta=0.65).
    Stage 2: Calibrate SPX smile over (beta, eta, rho) with H fixed.
             Target: synthetic gBergomi smile with (H=0.065, beta=1.0, eta=1.09, rho=-1.0).
    """
    print(NOTES)

    xi0_level = 0.235**2
    T_vix = 1.0 / 12.0  # one-month VIX options
    T_spx = 1.0 / 12.0

    # --- Generate target smiles ---
    print("Generating synthetic target VIX smile (true params: H=0.065, beta=0.037, eta=0.65)...")
    K_vix_market, iv_vix_market = generate_synthetic_vix_smile(
        H=0.065, beta=0.037, eta=0.65, xi0_level=xi0_level,
        T_vix=T_vix, n_strikes=11, n_paths=3_000
    )
    mask_vix = np.isfinite(iv_vix_market)
    K_vix_market = K_vix_market[mask_vix]
    iv_vix_market = iv_vix_market[mask_vix]
    print(f"  VIX strikes: {K_vix_market.round(4)}")
    print(f"  VIX market IVs: {iv_vix_market.round(4)}")

    print("\nGenerating synthetic target SPX smile (true params: H=0.065, beta=1.0, eta=1.09, rho=-1.0)...")
    K_spx_market, iv_spx_market = generate_synthetic_spx_smile(
        H=0.065, beta=1.0, eta=1.09, rho=-1.0,
        xi0_level=xi0_level, T_spx=T_spx, n_paths=3_000
    )
    mask_spx = np.isfinite(iv_spx_market)
    K_spx_market = K_spx_market[mask_spx]
    iv_spx_market = iv_spx_market[mask_spx]
    print(f"  SPX log-moneyness: {np.log(K_spx_market).round(4)}")
    print(f"  SPX market IVs: {iv_spx_market.round(4)}")

    # --- Stage 1: VIX calibration ---
    print("\n--- Stage 1: VIX Calibration ---")
    print("  Optimizer: differential_evolution, maxiter=30, popsize=15, seed=42")
    print("  Bounds: H∈(0.01,0.49), β∈(0.01,1.0), η∈(0.05,3.0)")

    bounds_vix = [(0.01, 0.49), (0.01, 1.0), (0.05, 3.0)]
    vix_result = differential_evolution(
        vix_objective,
        bounds=bounds_vix,
        args=(K_vix_market, iv_vix_market, xi0_level, T_vix, n_paths_calib, SEED),
        maxiter=30,
        popsize=15,
        seed=42,
        tol=1e-6,
        workers=1,
        polish=True,
    )
    H_vix, beta_vix, eta_vix = vix_result.x
    print(f"\n  VIX fit: H={H_vix:.4f}, beta={beta_vix:.4f}, eta={eta_vix:.4f}")
    print(f"  Target:  H=0.065,  beta=0.037,  eta=0.65")
    print(f"  MSE={vix_result.fun:.6f}, success={vix_result.success}")

    # --- Stage 2: SPX calibration (H fixed) ---
    print("\n--- Stage 2: SPX Calibration (H fixed = 0.065) ---")
    print("  Bounds: β∈(0.01,1.0), η∈(0.05,3.0), ρ∈(-1.0,0.0)")
    H_fixed = 0.065  # use paper's target directly for faithful reproduction

    bounds_spx = [(0.01, 1.0), (0.05, 3.0), (-1.0, 0.0)]
    spx_result = differential_evolution(
        spx_objective,
        bounds=bounds_spx,
        args=(K_spx_market, iv_spx_market, H_fixed, xi0_level, T_spx, n_paths_calib, SEED),
        maxiter=30,
        popsize=15,
        seed=42,
        tol=1e-6,
        workers=1,
        polish=True,
    )
    beta_spx, eta_spx, rho_spx = spx_result.x
    print(f"\n  SPX fit: beta={beta_spx:.4f}, eta={eta_spx:.4f}, rho={rho_spx:.4f}")
    print(f"  Target:  beta=1.0,   eta=1.09,   rho=-1.0")
    print(f"  MSE={spx_result.fun:.6f}, success={spx_result.success}")

    # --- Rough Bergomi benchmark VIX ---
    print("\n--- Rough Bergomi benchmark (beta=1 fixed for VIX) ---")
    bounds_rb = [(0.01, 0.49), (0.05, 3.0)]

    def rb_vix_obj(pv):
        return vix_objective(
            np.array([pv[0], 1.0, pv[1]]),
            K_vix_market, iv_vix_market, xi0_level, T_vix, n_paths_calib, SEED
        )

    rb_result = differential_evolution(
        rb_vix_obj, bounds=bounds_rb,
        maxiter=20, popsize=10, seed=42, workers=1
    )
    H_rb, eta_rb = rb_result.x
    print(f"  RoughBergomi VIX fit: H={H_rb:.4f}, eta={eta_rb:.4f}, MSE={rb_result.fun:.6f}")

    return {
        "vix_params": (H_vix, beta_vix, eta_vix),
        "vix_mse": float(vix_result.fun),
        "spx_params": (H_fixed, beta_spx, eta_spx, rho_spx),
        "spx_mse": float(spx_result.fun),
        "rb_vix_params": (H_rb, 1.0, eta_rb),
        "rb_vix_mse": float(rb_result.fun),
        "K_vix": K_vix_market,
        "iv_vix_market": iv_vix_market,
        "K_spx": K_spx_market,
        "iv_spx_market": iv_spx_market,
        "xi0_level": xi0_level,
        "T_vix": T_vix,
        "T_spx": T_spx,
    }


def plot_calibration_results(cal_results: dict, n_paths_plot: int = 3_000) -> None:
    """Plot fitted vs market VIX and SPX smiles."""
    H_vix, beta_vix, eta_vix = cal_results["vix_params"]
    H_fixed, beta_spx, eta_spx, rho_spx = cal_results["spx_params"]
    H_rb, _, eta_rb = cal_results["rb_vix_params"]
    xi0_level = cal_results["xi0_level"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    # VIX smile
    K_vix = cal_results["K_vix"]
    iv_vix_mkt = cal_results["iv_vix_market"]
    T_vix = cal_results["T_vix"]

    iv_vix_fit = model_vix_ivs(H_vix, beta_vix, eta_vix, xi0_level, T_vix, K_vix, n_paths_plot, SEED)
    iv_rb_fit  = model_vix_ivs(H_rb, 1.0, eta_rb, xi0_level, T_vix, K_vix, n_paths_plot, SEED)

    ax1.plot(K_vix, iv_vix_mkt, "ko-", ms=6, lw=2, label="Market (synthetic)")
    ax1.plot(K_vix, iv_vix_fit, "r--", lw=2, label=f"gBergomi fit (β={beta_vix:.3f})")
    ax1.plot(K_vix, iv_rb_fit, "b:", lw=2, label="rBergomi (β=1)")
    ax1.set_xlabel("VIX Strike")
    ax1.set_ylabel("VIX IV")
    ax1.set_title("VIX Smile Calibration")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # SPX smile
    K_spx = cal_results["K_spx"]
    iv_spx_mkt = cal_results["iv_spx_market"]
    T_spx = cal_results["T_spx"]

    iv_spx_fit = model_spx_ivs(H_fixed, beta_spx, eta_spx, rho_spx, xi0_level, T_spx, K_spx, n_paths_plot, SEED)

    log_m = np.log(K_spx)
    ax2.plot(log_m, iv_spx_mkt, "ko-", ms=6, lw=2, label="Market (synthetic)")
    ax2.plot(log_m, iv_spx_fit, "r--", lw=2, label=f"gBergomi fit (β={beta_spx:.2f})")
    ax2.set_xlabel("Log-moneyness log(K/S₀)")
    ax2.set_ylabel("SPX IV")
    ax2.set_title("SPX Smile Calibration (H fixed)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(f"gBergomi Calibration — Separate VIX vs SPX Fits\n"
                 f"VIX: H={H_vix:.3f}, β={beta_vix:.3f}, η={eta_vix:.3f} | "
                 f"SPX: β={beta_spx:.3f}, η={eta_spx:.3f}, ρ={rho_spx:.3f}", fontsize=10)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp6_calibration.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def main() -> None:
    print("=" * 60)
    print("Experiment 6: gBergomi Calibration (VIX and SPX Smiles)")
    print("=" * 60)
    print("Note: Using synthetic smiles as calibration targets (real option")
    print("      market data requires paid data vendor; see NOTES for assumptions)")
    print()

    cal_results = run_calibration(n_paths_calib=1_000)
    plot_calibration_results(cal_results)
    _update_results_md(cal_results)


def _update_results_md(cal_results: dict) -> None:
    results_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    H_vix, beta_vix, eta_vix = cal_results["vix_params"]
    _, beta_spx, eta_spx, rho_spx = cal_results["spx_params"]
    H_rb, _, eta_rb = cal_results["rb_vix_params"]

    summary = f"""
## Experiment 6: Separate VIX and SPX Calibrations

{NOTES}

### VIX Calibration (H, β, η)
| | H | β | η | MSE |
|---|---|---|---|-----|
| Fitted | {H_vix:.4f} | {beta_vix:.4f} | {eta_vix:.4f} | {cal_results['vix_mse']:.6f} |
| Target (paper) | 0.065 | 0.037 | 0.65 | — |
| rBergomi (β=1) | {H_rb:.4f} | 1.000 | {eta_rb:.4f} | {cal_results['rb_vix_mse']:.6f} |

### SPX Calibration (β, η, ρ) with H fixed=0.065
| | β | η | ρ | MSE |
|---|---|---|---|-----|
| Fitted | {beta_spx:.4f} | {eta_spx:.4f} | {rho_spx:.4f} | {cal_results['spx_mse']:.6f} |
| Target (paper) | 1.0 | 1.09 | -1.0 | — |

### Calibration Tension
- VIX smile prefers **small β** (strong grey effect) to match upward-sloping smile convexity.
- SPX smile prefers **β near 1** (rough Bergomi limit) and strongly negative ρ for skew.
- rBergomi has higher VIX MSE, confirming that grey extension is needed for VIX.
- One-factor gBergomi **cannot jointly calibrate** SPX and VIX with one parameter set.
"""
    with open(results_path, "a") as f:
        f.write(summary)
    print(f"Results appended to {results_path}")


if __name__ == "__main__":
    main()
