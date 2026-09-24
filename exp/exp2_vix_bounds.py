"""
Experiment 2: VIX Futures Bounds Under gBergomi Across Forward Variance Scenarios.

Reproduces and validates the analytic lower and upper bounds for VIX futures
stated in Proposition 2.15, across three forward variance curve scenarios.

Convention notes:
  - Delta = 1/12 (one calendar month)
  - Parameters: H=0.065, beta=0.037, eta=0.65 (paper Figure 1)
  - T_grid: [0.01, 2.0] in 100 steps
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    GBergomiParams, xi_scenario1, xi_scenario2, xi_scenario3, DELTA, SEED
)
from src.vix_bounds import compute_vix_bounds

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_vix_bounds_analysis() -> dict:
    """Compute and compare VIX bounds for all three xi_0 scenarios."""
    # Paper Figure 1 parameters
    params_grey = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
    params_rb = GBergomiParams(H=0.065, beta=1.0, eta=0.65)  # rough Bergomi benchmark

    T_grid = np.linspace(0.01, 2.0, 100)
    Delta = DELTA  # 1/12

    scenarios = {
        "Scenario 1: flat ξ₀(t)=0.235²": xi_scenario1,
        "Scenario 2: quadratic ξ₀(t)=0.235²(1+t)²": xi_scenario2,
        "Scenario 3: sqrt ξ₀(t)=0.235²√(1+t)": xi_scenario3,
    }

    results = {}
    print("Computing VIX bounds (Proposition 2.15)...")
    print(f"  Parameters: H={params_grey.H}, beta={params_grey.beta}, eta={params_grey.eta}")
    print(f"  Delta = {Delta:.6f} years ({Delta*365:.1f} days)")
    print(f"  T_grid: [{T_grid[0]:.3f}, {T_grid[-1]:.3f}] ({len(T_grid)} points)")

    for scenario_name, xi0_func in scenarios.items():
        print(f"\n  {scenario_name}")
        lb, ub = compute_vix_bounds(params_grey, xi0_func, T_grid, Delta)
        lb_rb, ub_rb = compute_vix_bounds(params_rb, xi0_func, T_grid, Delta)

        # Verify LB <= UB
        n_violations = np.sum(lb > ub + 1e-8)
        assert n_violations == 0, f"LB > UB violations: {n_violations}"
        print(f"    LB <= UB check: PASSED")
        print(f"    LB range: [{lb.min():.4f}, {lb.max():.4f}]")
        print(f"    UB range: [{ub.min():.4f}, {ub.max():.4f}]")

        results[scenario_name] = {
            "T_grid": T_grid,
            "lb": lb,
            "ub": ub,
            "lb_rb": lb_rb,
            "ub_rb": ub_rb,
        }

    return results


def plot_vix_bounds(results: dict) -> None:
    """Plot VIX bounds for all scenarios (Figure 1 style)."""
    n_scenarios = len(results)
    fig, axes = plt.subplots(1, n_scenarios, figsize=(14, 5))

    scenario_labels = list(results.keys())
    colors = {"grey": "#e74c3c", "rough": "#3498db"}

    for ax, scenario_name in zip(axes, scenario_labels):
        data = results[scenario_name]
        T = data["T_grid"]
        lb = data["lb"]
        ub = data["ub"]
        lb_rb = data["lb_rb"]

        ax.plot(T, ub, color="gray", lw=2, linestyle="--", label="UB (Jensen)")
        ax.plot(T, lb, color=colors["grey"], lw=2, label="LB (grey β=0.037)")
        ax.plot(T, lb_rb, color=colors["rough"], lw=1.5, linestyle=":", label="LB (rBergomi β=1)")
        ax.fill_between(T, lb, ub, alpha=0.1, color=colors["grey"])

        # Short readable title
        title = scenario_name.split(":")[0]
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("Maturity T (years)")
        ax.set_ylabel("VIX futures bound" if scenario_name == scenario_labels[0] else "")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("VIX Futures Bounds — Proposition 2.15 (H=0.065, β=0.037, η=0.65)",
                 fontsize=11)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "exp2_vix_bounds.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {path}")


def main() -> None:
    print("=" * 60)
    print("Experiment 2: VIX Futures Bounds Under gBergomi")
    print("=" * 60)

    results = run_vix_bounds_analysis()
    plot_vix_bounds(results)

    # Summary statistics
    print("\nSummary (Scenario 1 at T=0.1, 0.5, 1.0, 2.0):")
    s1 = results["Scenario 1: flat ξ₀(t)=0.235²"]
    for T_val in [0.1, 0.5, 1.0, 2.0]:
        idx = np.argmin(np.abs(s1["T_grid"] - T_val))
        print(f"  T={T_val:.1f}: LB={s1['lb'][idx]:.4f}, UB={s1['ub'][idx]:.4f}, "
              f"gap={s1['ub'][idx]-s1['lb'][idx]:.4f}")

    # Update RESULTS.md
    _update_results_md(results)


def _update_results_md(results: dict) -> None:
    results_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    s1 = results["Scenario 1: flat ξ₀(t)=0.235²"]
    summary = f"""
## Experiment 2: VIX Futures Bounds (Proposition 2.15)

**Parameters**: H=0.065, β=0.037, η=0.65, Δ=1/12 year  
**Scenarios**: Flat, quadratic, and sqrt forward-variance curves  
**T range**: [0.01, 2.0]

### Scenario 1 (Flat xi_0) Key Values
| T    | LB (grey) | UB (Jensen) | Gap  |
|------|-----------|-------------|------|
| 0.10 | {s1['lb'][np.argmin(np.abs(s1['T_grid']-0.1))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-0.1))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-0.1))]-s1['lb'][np.argmin(np.abs(s1['T_grid']-0.1))]:.4f} |
| 0.50 | {s1['lb'][np.argmin(np.abs(s1['T_grid']-0.5))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-0.5))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-0.5))]-s1['lb'][np.argmin(np.abs(s1['T_grid']-0.5))]:.4f} |
| 1.00 | {s1['lb'][np.argmin(np.abs(s1['T_grid']-1.0))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-1.0))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-1.0))]-s1['lb'][np.argmin(np.abs(s1['T_grid']-1.0))]:.4f} |
| 2.00 | {s1['lb'][np.argmin(np.abs(s1['T_grid']-2.0))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-2.0))]:.4f} | {s1['ub'][np.argmin(np.abs(s1['T_grid']-2.0))]-s1['lb'][np.argmin(np.abs(s1['T_grid']-2.0))]:.4f} |

**Note**: UB is constant for flat xi_0 (= sqrt(xi_0) = 0.2350). LB < UB for all T.  
LB increases with T for small beta due to Mittag-Leffler normalization decreasing.  
"""
    with open(results_path, "a") as f:
        f.write(summary)
    print(f"Results appended to {results_path}")


if __name__ == "__main__":
    main()
