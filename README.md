# Grey Bergomi Model

A Python implementation of the **grey Bergomi (gBergomi) model** — a generalisation of the rough Bergomi model that replaces the Gaussian stochastic variance driver with a grey Brownian motion constructed from the M-Wright distribution and the Mittag-Leffler function.

---

## Mathematical Background

### Rough Bergomi recap

The rough Bergomi model (El Euch & Rosenbaum, 2019) drives instantaneous variance with a Riemann–Liouville fractional Brownian motion:

```
V_t = xi_0(t) * exp( eta * W^H_t - (1/2) * eta^2 * t^{2H} )
```

where `H ∈ (0, 1/2)` is the Hurst exponent and `xi_0` is the initial forward-variance curve.

### Grey Bergomi extension

The grey Bergomi model introduces a second dimension of randomness via the M-Wright random variable `Y_beta`:

```
V_t = xi_0(t) * Y_beta * exp( eta * M^H_t - (1/2) * eta^2 * E_beta(b * t^{2H}) )
```

where:

| Symbol | Meaning |
|--------|---------|
| `beta ∈ (0, 1]` | Mittag-Leffler / M-Wright index |
| `Y_beta` | M-Wright distributed random variable (independent across paths) |
| `M^H_t` | Gaussian process with covariance kernel involving `t^{2H}` |
| `E_beta(z)` | One-parameter Mittag-Leffler function |
| `b = eta^2` | Kernel scaling coefficient |

Setting `beta = 1` recovers the standard rough Bergomi model exactly (`Y_1 = 1` a.s.).

### Key mathematical objects

**Mittag-Leffler function**

```
E_beta(z) = sum_{k=0}^{inf} z^k / Gamma(beta*k + 1)
```

- `E_1(z) = exp(z)`
- `E_beta(0) = 1` for all `beta`
- `E_beta(z) > 1` for `z > 0`

**M-Wright distribution**

The random variable `Y_beta` follows the M-Wright (or Mainardi) distribution with index `beta`. Its moments are:

```
E[Y_beta^kappa] = Gamma(kappa + 1) / Gamma(beta*kappa + 1)
```

For `beta = 1`, `Y_1 = 1` deterministically.

---

## Project Structure

```
project/
├── src/                        # Core library
│   ├── __init__.py
│   ├── config.py               # GBergomiParams dataclass + forward-variance scenarios
│   ├── special_functions.py    # Mittag-Leffler function and grid builder
│   ├── mwright.py              # M-Wright sampler and theoretical moments
│   ├── gbergomi.py             # Variance path simulation and martingale check
│   └── vix_bounds.py           # Analytical VIX futures upper/lower bounds
│
├── tests/                      # Pytest test suite
│   ├── __init__.py
│   ├── test_special_functions.py
│   ├── test_mwright.py
│   ├── test_gbergomi.py
│   └── test_vix_bounds.py
│
├── exp/                        # Experiment scripts
│   ├── __init__.py
│   ├── exp1_building_blocks.py
│   ├── exp2_vix_bounds.py
│   ├── exp3_vix_mc.py
│   ├── exp4_spx_mc.py
│   ├── exp5_asymptotics.py
│   ├── exp6_calibration.py
│   ├── exp7_benchmark.py
│   └── exp8_vix_empirical.py
│
├── data/                       # Market data (gitignored)
├── results/
│   └── RESULTS.md              # Experiment output summary
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repo-url>
cd project
```

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `py_vollib` and `py_vollib_vectorized` require `TA-Lib` headers on some platforms.
> On Ubuntu/Debian: `sudo apt-get install libta-lib-dev`

---

## Running Tests

```bash
# Run the full test suite from the project root
pytest tests/ -v

# Run a specific test file
pytest tests/test_special_functions.py -v
pytest tests/test_mwright.py -v
pytest tests/test_gbergomi.py -v
pytest tests/test_vix_bounds.py -v
```

Tests for `src.vix_bounds` gracefully skip if that module has not yet been implemented.

---

## Running Experiments

Each experiment script is self-contained and writes figures to `results/`.

```bash
# Experiment 1 – Core building blocks: Mittag-Leffler, M-Wright, variance paths
python exp/exp1_building_blocks.py

# Experiment 2 – Analytical VIX futures bounds
python exp/exp2_vix_bounds.py

# Experiment 3 – Monte Carlo VIX futures prices
python exp/exp3_vix_mc.py

# Experiment 4 – SPX option pricing via Monte Carlo
python exp/exp4_spx_mc.py

# Experiment 5 – Short-maturity implied-vol asymptotics
python exp/exp5_asymptotics.py

# Experiment 6 – Joint SPX/VIX market calibration
python exp/exp6_calibration.py

# Experiment 7 – Benchmark against rough Bergomi (beta=1)
python exp/exp7_benchmark.py

# Experiment 8 – Empirical VIX term-structure analysis
python exp/exp8_vix_empirical.py
```

Results (figures and metrics) are documented in [`results/RESULTS.md`](results/RESULTS.md).

---

## Key Model Parameters

| Parameter | Typical range | Description |
|-----------|--------------|-------------|
| `H` | 0.05 – 0.2 | Hurst exponent (roughness) |
| `beta` | 0.037 – 1.0 | M-Wright index (`beta=1` → rough Bergomi) |
| `eta` | 0.5 – 2.0 | Vol-of-vol |

### Baseline calibration (Scenario 1)

```python
from src.config import GBergomiParams, xi_scenario1

params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
# Forward-variance curve: flat at xi_0 = 0.04 (20% implied vol)
```

---

## Module Reference

### `src.config`

```python
GBergomiParams(H, beta, eta)   # Dataclass; b = eta**2 derived automatically
xi_scenario1(t)                 # Flat forward-variance curve: xi_0(t) = 0.04
```

### `src.special_functions`

```python
mittag_leffler_stable(z, beta)              # Scalar E_beta(z) via mpmath / series
build_mittag_leffler_grid(t_grid, b, beta, H)  # Vectorised E_beta(b * t^{2H}) array
```

### `src.mwright`

```python
sample_mwright(n, beta, rng)        # Draw n samples from M-Wright(beta) distribution
theoretical_moment(kappa, beta)     # E[Y_beta^kappa] = Gamma(kappa+1)/Gamma(beta*kappa+1)
```

### `src.gbergomi`

```python
simulate_variance_paths(params, xi_0, t_grid, n_paths, rng)
# Returns dict with keys:
#   'V'      : variance paths, shape (n_paths, n_t)
#   'Y_beta' : M-Wright draws,  shape (n_paths,)

check_martingale_condition(V, t_grid, xi_0)
# Returns array of ratios E[V_t] / xi_0(t), shape (n_t,)
```

### `src.vix_bounds`

```python
compute_vix_bounds(params, xi_0, T_grid, Delta)
# Returns (lb, ub) arrays of shape (len(T_grid),)
# where lb = Jensen lower bound, ub = sqrt(xi_0) upper bound
```

---

## References

1. El Euch, O. & Rosenbaum, M. (2019). *The characteristic function of rough Heston models.* Mathematical Finance.
2. Mainardi, F., Mura, A. & Pagnini, G. (2010). *The M-Wright function in time-fractional diffusion processes.* International Journal of Differential Equations.
3. Gorenflo, R., Kilbas, A., Mainardi, F. & Rogosin, S. (2014). *Mittag-Leffler Functions, Related Topics and Applications.* Springer.
4. Bayer, C., Friz, P. & Gatheral, J. (2016). *Pricing under rough volatility.* Quantitative Finance.
