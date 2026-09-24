import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def test_y_beta_is_1_when_beta1():
    import numpy as np
    from src.mwright import sample_mwright
    rng = np.random.default_rng(42)
    samples = sample_mwright(1000, 1.0, rng)
    assert np.allclose(samples, 1.0), 'Y_1 should be 1 for all paths'

def test_mwright_positive():
    import numpy as np
    from src.mwright import sample_mwright
    for beta in [0.1, 0.3, 0.5, 0.8]:
        rng = np.random.default_rng(42)
        samples = sample_mwright(10000, beta, rng)
        assert np.all(samples > 0), f'All Y_beta samples should be positive for beta={beta}'

def test_mwright_moment_beta05():
    import numpy as np
    from src.mwright import sample_mwright, theoretical_moment
    rng = np.random.default_rng(42)
    beta = 0.5
    n = 200000
    samples = sample_mwright(n, beta, rng)
    for kappa in [0.25, 0.5, 1.0]:
        emp = np.mean(samples**kappa)
        theo = theoretical_moment(kappa, beta)
        rel_err = abs(emp - theo) / theo
        assert rel_err < 0.1, f'kappa={kappa}: emp={emp:.4f} theo={theo:.4f} rel_err={rel_err:.4f}'

def test_theoretical_moment_kappa0():
    from src.mwright import theoretical_moment
    import numpy as np
    # E[Y_beta^0] = 1 for all beta
    for beta in [0.1, 0.5, 1.0]:
        assert abs(theoretical_moment(0.0, beta) - 1.0) < 1e-10

def test_theoretical_moment_kappa1_beta1():
    from src.mwright import theoretical_moment
    # E[Y_1^1] = Gamma(2)/Gamma(2) = 1
    assert abs(theoretical_moment(1.0, 1.0) - 1.0) < 1e-10
