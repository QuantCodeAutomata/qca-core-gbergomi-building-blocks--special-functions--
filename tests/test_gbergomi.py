import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_variance_martingale_condition():
    import numpy as np
    from src.config import GBergomiParams, xi_scenario1
    from src.gbergomi import simulate_variance_paths, check_martingale_condition
    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    t_grid = np.array([0.1, 0.25, 0.5, 1.0])
    rng = np.random.default_rng(42)
    result = simulate_variance_paths(params, xi_scenario1, t_grid, n_paths=5000, rng=rng)
    ratios = check_martingale_condition(result['V'], t_grid, xi_scenario1)
    # E[V_t] / xi_0(t) should be close to 1
    assert np.all(np.abs(ratios - 1.0) < 0.05), f'Martingale ratios: {ratios}'

def test_variance_positive():
    import numpy as np
    from src.config import GBergomiParams, xi_scenario1
    from src.gbergomi import simulate_variance_paths
    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    t_grid = np.array([0.1, 0.5, 1.0])
    rng = np.random.default_rng(42)
    result = simulate_variance_paths(params, xi_scenario1, t_grid, n_paths=1000, rng=rng)
    assert np.all(result['V'] > 0), 'All variance values should be positive'

def test_beta1_reduces_to_rbergomi():
    import numpy as np
    from src.config import GBergomiParams, xi_scenario1
    from src.gbergomi import simulate_variance_paths
    params = GBergomiParams(H=0.065, beta=1.0, eta=0.65)
    t_grid = np.array([0.1, 0.5, 1.0])
    rng = np.random.default_rng(42)
    result = simulate_variance_paths(params, xi_scenario1, t_grid, n_paths=1000, rng=rng)
    # Y_beta should be 1 for beta=1
    assert np.allclose(result['Y_beta'], 1.0), 'Y_1 should be exactly 1'

def test_variance_shape():
    import numpy as np
    from src.config import GBergomiParams, xi_scenario1
    from src.gbergomi import simulate_variance_paths
    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    n_t, n_paths = 10, 500
    t_grid = np.linspace(0.1, 1.0, n_t)
    rng = np.random.default_rng(0)
    result = simulate_variance_paths(params, xi_scenario1, t_grid, n_paths=n_paths, rng=rng)
    assert result['V'].shape == (n_paths, n_t)
    assert result['Y_beta'].shape == (n_paths,)
