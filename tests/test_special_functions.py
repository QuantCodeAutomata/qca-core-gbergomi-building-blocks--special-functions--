import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_mittag_leffler_beta1():
    # E_1(z) should equal exp(z)
    from src.special_functions import mittag_leffler_stable
    import numpy as np
    for z in [0.0, 0.5, 1.0, 2.0]:
        assert abs(mittag_leffler_stable(z, 1.0) - np.exp(z)) < 1e-6, f'Failed at z={z}'

def test_mittag_leffler_at_zero():
    from src.special_functions import mittag_leffler_stable
    # E_beta(0) = 1 for all beta
    for beta in [0.037, 0.1, 0.5, 1.0]:
        assert abs(mittag_leffler_stable(0.0, beta) - 1.0) < 1e-10

def test_mittag_leffler_positive():
    from src.special_functions import mittag_leffler_stable
    # E_beta(z) > 1 for z > 0
    for beta in [0.1, 0.5, 0.8, 1.0]:
        for z in [0.1, 0.5, 1.0]:
            val = mittag_leffler_stable(z, beta)
            assert val > 1.0, f'E_{beta}({z}) = {val} should be > 1'

def test_build_mittag_leffler_grid():
    from src.special_functions import build_mittag_leffler_grid
    from src.config import GBergomiParams
    import numpy as np
    params = GBergomiParams(H=0.065, beta=0.5, eta=0.65)
    t_grid = np.linspace(0.01, 1, 10)
    result = build_mittag_leffler_grid(t_grid, params.b, params.beta, params.H)
    assert result.shape == (10,)
    assert np.all(result >= 1.0)
