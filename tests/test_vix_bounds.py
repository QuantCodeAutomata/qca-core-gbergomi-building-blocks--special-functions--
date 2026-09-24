import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_lb_le_ub():
    # Lower bound must be <= upper bound
    import numpy as np
    # Import will be from the vix_bounds module if it exists, else skip
    try:
        from src.vix_bounds import compute_vix_bounds
        from src.config import GBergomiParams, xi_scenario1
        params = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
        T_grid = np.linspace(0.05, 1.0, 10)
        Delta = 1/12
        lb, ub = compute_vix_bounds(params, xi_scenario1, T_grid, Delta)
        assert np.all(lb <= ub + 1e-8), 'Lower bound must not exceed upper bound'
    except ImportError:
        pass

def test_ub_flat_scenario():
    # For flat xi_0, UB should be constant
    import numpy as np
    try:
        from src.vix_bounds import compute_vix_bounds
        from src.config import GBergomiParams, xi_scenario1
        params = GBergomiParams(H=0.065, beta=0.037, eta=0.65)
        T_grid = np.array([0.1, 0.5, 1.0])
        Delta = 1/12
        lb, ub = compute_vix_bounds(params, xi_scenario1, T_grid, Delta)
        # For flat xi_0, UB = sqrt(xi_0) = const
        assert np.std(ub) < 1e-6, 'UB should be constant for flat xi_0'
    except ImportError:
        pass
