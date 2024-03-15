import pytest

import numpy as np
from scipy.integrate import quad, dblquad  # type: ignore

from tmvbeta import TBeta, TMVBeta, TMVNormal


class TestTBeta:
    def test_tbeta_rvs(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        # Test
        for a, b in zip(a_vals, b_vals):
            x = TBeta(a, b).rvs(size=int(1e6))
            np.testing.assert_allclose(np.mean(x), a / (a + b), atol=0.01)
            np.testing.assert_allclose(np.var(x), a * b / ((a + b)**2 * (a + b + 1)), atol=0.01)
            
    def test_tbeta_pdf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_max = 0.2, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            p_support, err_support = quad(tbeta.pdf, x_min, x_max)
            p_unitint, err_unitint = quad(tbeta.pdf, 0, 1)
            np.testing.assert_allclose(p_support, 1.0, atol=10 * err_support)
            np.testing.assert_allclose(p_unitint, 1.0, atol=10 * err_unitint)

    def test_tbeta_logpdf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_mid, x_max = 0.2, 0.5, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            np.testing.assert_allclose(tbeta.logpdf(x_mid),  np.log(tbeta.pdf(x_mid)))

    def test_tbeta_cdf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_max = 0.2, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            np.testing.assert_allclose(tbeta.cdf(0), 0.0)
            np.testing.assert_allclose(tbeta.cdf(x_min), 0.0)
            np.testing.assert_allclose(tbeta.cdf(x_max), 1.0)
            np.testing.assert_allclose(tbeta.cdf(1), 1.0)

    def test_tbeta_logcdf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_mid, x_max = 0.2, 0.5, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            np.testing.assert_allclose(tbeta.logcdf(x_mid),  np.log(tbeta.cdf(x_mid)))

    def test_tbeta_sf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_max = 0.2, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            np.testing.assert_allclose(tbeta.sf(0.0), 1.0)
            np.testing.assert_allclose(tbeta.sf(x_min), 1.0)
            np.testing.assert_allclose(tbeta.sf(x_max), 0.0)
            np.testing.assert_allclose(tbeta.sf(1.0), 0.0)

    def test_tbeta_logsf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_mid, x_max = 0.2, 0.5, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            np.testing.assert_allclose(tbeta.logsf(x_mid),  np.log(tbeta.sf(x_mid)))

    def test_tbeta_ppf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_max = 0.2, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            np.testing.assert_allclose(tbeta.ppf(0), x_min)
            np.testing.assert_allclose(tbeta.ppf(1), x_max)

    def test_tbeta_isf(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_max = 0.2, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            np.testing.assert_allclose(tbeta.isf(0.0), x_max)
            np.testing.assert_allclose(tbeta.isf(1.0), x_min)

    def test_tbeta_moment(self) -> None:
        # Data
        a_vals = [3, 3, 0.8, 0.8]
        b_vals = [2, 0.5, 2, 0.5]
        x_min, x_max = 0.2, 0.8
        # Test
        for a, b in zip(a_vals, b_vals):
            tbeta = TBeta(a, b, x_min, x_max)
            x = tbeta.rvs(size=int(1e6))
            for n in [1, 2, 3]:
                np.testing.assert_allclose(tbeta.moment(n), np.mean(x ** n), atol=0.1)
    
    def test_tbeta_fit(self) -> None:
        # Data
        x_1 = np.array([0.1, 0.9] + 7 * [0.5])
        x_2 = np.array([0.2, 0.8] + 7 * [0.5])
        x_min, x_max = 0.2, 0.8
        a_exp = b_exp = 3.566
        # Test
        tbeta = TBeta(1, 1, x_min, x_max)
        with pytest.raises(ValueError):
            tbeta.fit(x_1)
        tbeta.fit(x_2)
        np.testing.assert_allclose(tbeta.a, a_exp, atol=0.01)
        np.testing.assert_allclose(tbeta.b, b_exp, atol=0.01)

    def test_tbeta_fit_multi_sample(self) -> None:
        # Data
        x_1 = np.array([0.1, 0.9] + 7 * [0.5])
        x_2 = np.array([0.2, 0.8] + 7 * [0.5])
        X_min = [0.1, 0.2]
        X_max = [0.9, 0.8]
        a_exp = b_exp = 2.027
        # Test
        tbeta = TBeta(1, 1)
        with pytest.raises(ValueError):
            tbeta.fit_multi_sample([x_2, x_1], X_min, X_max)
        tbeta.fit_multi_sample([x_1, x_2], X_min, X_max)
        np.testing.assert_allclose(tbeta.a, a_exp, atol=0.01)
        np.testing.assert_allclose(tbeta.b, b_exp, atol=0.01)


class TestTMVNormal:
    def test_tmvnormal_rvs(self) -> None:
        # Data
        mean = np.array([-0.1, 0.1])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Test
        Z = TMVNormal(mean, cov).rvs(size=int(1e6))
        np.testing.assert_allclose(np.mean(Z, axis=0), mean, atol=0.01)
        np.testing.assert_allclose(np.cov(Z.T), cov, atol=0.01)

    def test_tmvnormal_pdf(self) -> None:
        # Data
        mean = np.array([-0.1, 0.1])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Test
        tmvnormal = TMVNormal(mean, cov, -np.ones(2), np.ones(2))
        p_support, err_support = dblquad(lambda x, y: tmvnormal.pdf(np.array([x, y])), -1, 1, -1, 1)
        np.testing.assert_allclose(p_support, 1.0, atol=10 * err_support)

    def test_tmvnormal_logpdf(self) -> None:
        # Data
        mean = np.array([-0.1, 0.1])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Test
        tmvnormal = TMVNormal(mean, cov, -np.ones(2), np.ones(2))
        np.testing.assert_allclose(tmvnormal.logpdf(np.zeros(2)),  np.log(tmvnormal.pdf(np.zeros(2))))

    def test_tmvnormal_cdf(self) -> None:
        # Data
        mean = np.array([-0.1, 0.1])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Test
        tmvnormal = TMVNormal(mean, cov, -np.ones(2), np.ones(2))
        np.testing.assert_allclose(tmvnormal.cdf(-np.ones(2)), 0.0)
        np.testing.assert_allclose(tmvnormal.cdf(np.ones(2)), 1.0)

    def test_tmvnormal_logcdf(self) -> None:
        # Data
        mean = np.array([-0.1, 0.1])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Test
        tmvnormal = TMVNormal(mean, cov, -np.ones(2), np.ones(2))
        np.testing.assert_allclose(tmvnormal.logcdf(np.zeros(2)),  np.log(tmvnormal.cdf(np.zeros(2))))
    
    def test_tmvnormal_fit(self) -> None:
        # Data
        mean_exp = np.zeros(2)
        cov_exp = np.array([[1.68251539e+00, 1.04291217e-05], [1.04291217e-05, 1.68251810e+00]])
        Z_1 = np.array([[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]] + 9 * [[0.0, 0.0]])
        Z_2 = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]] + 9 * [[0.0, 0.0]])
        # Test
        tmvnormal = TMVNormal(np.ones(2), np.eye(2), -np.ones(2), np.ones(2))
        with pytest.raises(ValueError):
            tmvnormal.fit(Z_1)
        tmvnormal.fit(Z_2)
        np.testing.assert_allclose(tmvnormal.mean, mean_exp, atol=0.01)
        np.testing.assert_allclose(tmvnormal.cov, cov_exp, atol=0.01)

    def test_tmvnormal_fit_multi_sample(self) -> None:
        # Data
        mean_exp = np.array([7.15541753e-06, 3.27370414e-06])
        cov_exp = np.array([[ 5.69722103e+00, -3.89149843e-06], [-3.89149843e-06,  5.69719264e+00]])
        Z_1 = np.array([[-2.0, -2.0], [2.0, 2.0], [-2.0, 2.0], [2.0, -2.0]] + 9 * [[0.0, 0.0]])
        Z_2 = np.array([[-1.0, -1.0], [1.0, 1.0], [-1.0, 1.0], [1.0, -1.0]] + 9 * [[0.0, 0.0]])
        Z_min = [np.full(2, -2.0), np.full(2, -1.0)] 
        Z_max = [np.full(2, 2.0), np.full(2, 1.0)]
        # Test
        tmvnormal = TMVNormal(np.zeros(2), np.eye(2))
        with pytest.raises(ValueError):
            tmvnormal.fit_multi_sample([Z_2, Z_1], Z_min, Z_max)
        tmvnormal.fit_multi_sample([Z_1, Z_2], Z_min, Z_max)
        np.testing.assert_allclose(tmvnormal.mean, mean_exp, atol=0.01)
        np.testing.assert_allclose(tmvnormal.cov, cov_exp, atol=0.01)


class TestTMVBeta:
    def test_tmvbeta_rvs(self) -> None:
        # Data
        a = np.array([4, 3])
        b = np.array([2, 5])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        # Test
        tmvbeta = TMVBeta(a, b, cov)
        X = tmvbeta.rvs(size=int(1e6))
        np.testing.assert_allclose(np.mean(X, axis=0), a / (a + b), atol=0.01)
        np.testing.assert_allclose(np.cov(tmvbeta.mvbeta._X_to_Z(np.array(X)).T), cov, atol=0.01)

    def test_tmvbeta_pdf(self) -> None:
        # Data
        a = np.array([4, 3])
        b = np.array([2, 5])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        x_min, x_max = 0.2, 0.8
        # Test
        tmvbeta = TMVBeta(a, b, cov, np.full(2, x_min), np.full(2, x_max))
        p_support, err_support = dblquad(lambda x, y: tmvbeta.pdf(np.array([x, y])), x_min, x_max, x_min, x_max)
        np.testing.assert_allclose(p_support, 1.0, atol=10 * err_support)

    def test_tmvbeta_logpdf(self) -> None:
        # Data
        a = np.array([4, 3])
        b = np.array([2, 5])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        x_min, x_mid, x_max = 0.2, 0.5, 0.8
        # Test
        tmvbeta = TMVBeta(a, b, cov, np.full(2, x_min), np.full(2, x_max))
        np.testing.assert_allclose(tmvbeta.logpdf(np.full(2, x_mid)),  np.log(tmvbeta.pdf(np.full(2, x_mid))))

    def test_tmvbeta_cdf(self) -> None:
        # Data
        a = np.array([4, 3])
        b = np.array([2, 5])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        x_min, x_max = 0.2, 0.8
        # Test
        tmvbeta = TMVBeta(a, b, cov, np.full(2, x_min), np.full(2, x_max))
        np.testing.assert_allclose(tmvbeta.cdf(np.full(2, x_min)), 0.0)
        np.testing.assert_allclose(tmvbeta.cdf(np.full(2, x_max)), 1.0)

    def test_tmvbeta_logcdf(self) -> None:
        # Data
        a = np.array([4, 3])
        b = np.array([2, 5])
        cov = np.array([[1.0, 0.5], [0.5, 1.0]])
        x_min, x_mid, x_max = 0.2, 0.5, 0.8
        # Test
        tmvbeta = TMVBeta(a, b, cov, np.full(2, x_min), np.full(2, x_max))
        np.testing.assert_allclose(tmvbeta.logcdf(np.full(2, x_mid)),  np.log(tmvbeta.cdf(np.full(2, x_mid))))

    def test_tmvbeta_fit_marginal(self) -> None:
        # Data
        X_1 = np.array([[0.2, 0.2], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]] + 12 * [[0.5, 0.5]])
        X_2 = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        x_min, x_max = 0.2, 0.8
        a_exp = 4.063
        b_exp = 3.891
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2), x_min=np.full(2, x_min), x_max=np.full(2, x_max))
        with pytest.raises(ValueError):
            tmvbeta.fit_marginals(X_1)
        tmvbeta.fit_marginals(X_2)
        np.testing.assert_allclose(tmvbeta.a, a_exp, atol=0.01)
        np.testing.assert_allclose(tmvbeta.b, b_exp, atol=0.01)
    
    def test_tmvbeta_fit_cov(self) -> None:
        # Data
        X_1 = np.array([[0.2, 0.2], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]] + 12 * [[0.5, 0.5]])
        X_2 = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        x_min, x_max = 0.2, 0.8
        cov_exp = np.array([[1.0, -0.6], [-0.6, 1.0]])
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2), x_min=np.full(2, x_min), x_max=np.full(2, x_max))
        with pytest.raises(ValueError):
            tmvbeta.fit_cov(X_1)
        tmvbeta.fit_cov(X_2)
        np.testing.assert_allclose(tmvbeta.cov, cov_exp, atol=0.01)
       
    def test_tmvbeta_fit(self) -> None:
        # Data
        X_1 = np.array([[0.2, 0.2], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]] + 12 * [[0.5, 0.5]])
        X_2 = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        x_min, x_max = 0.2, 0.8
        a_exp = np.array([3.98117738, 3.98118726])
        b_exp = np.array([3.82437605, 3.8243916 ])
        cov_exp = np.array([[ 1.0, -0.25544013], [-0.25544013,  1.0]])
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2), x_min=np.full(2, x_min), x_max=np.full(2, x_max))
        with pytest.raises(ValueError):
            tmvbeta.fit(X_1)
        tmvbeta.fit(X_2)
        np.testing.assert_allclose(tmvbeta.a, a_exp, atol=0.01)
        np.testing.assert_allclose(tmvbeta.b, b_exp, atol=0.01)
        np.testing.assert_allclose(tmvbeta.cov, cov_exp, atol=0.01)

    def test_tmvbeta_fit_mode(self) -> None:
        # Data
        X = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        mode_lb = np.full(2, 0.6)
        mode_ub = np.full(2, 0.4)
        x_min, x_max = 0.2, 0.8
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2), x_min=np.full(2, x_min), x_max=np.full(2, x_max))
        tmvbeta.fit(X, mode_lb=mode_lb)
        assert np.all((tmvbeta.a - 1) / (tmvbeta.a + tmvbeta.b - 2) >= mode_lb - 1e-4)
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2))
        tmvbeta.fit(X, mode_ub=mode_ub)
        assert np.all((tmvbeta.a - 1) / (tmvbeta.a + tmvbeta.b - 2) <= mode_ub + 1e-4)

    def test_tmvbeta_fit_marginal_multi_sample(self) -> None:
        # Data
        X_1 = np.array([[0.2, 0.2], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]] + 12 * [[0.5, 0.5]])
        X_2 = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        X_min = [np.full(2, 0.1), np.full(2, 0.2)]
        X_max = [np.full(2, 0.9), np.full(2, 0.8)]
        a_exp = np.array([2.27967692, 2.27967692])
        b_exp = np.array([2.15374755, 2.15374755])
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2))
        with pytest.raises(ValueError):
            tmvbeta.fit_marginals_multi_sample([X_2, X_1], X_min, X_max)
        tmvbeta.fit_marginals_multi_sample([X_1, X_2], X_min, X_max)
        np.testing.assert_allclose(tmvbeta.a, a_exp, atol=0.01)
        np.testing.assert_allclose(tmvbeta.b, b_exp, atol=0.01)

    def test_tmvbeta_fit_cov_multi_sample(self) -> None:
        # Data
        X_1 = np.array([[0.2, 0.2], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]] + 12 * [[0.5, 0.5]])
        X_2 = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        X_min = [np.full(2, 0.1), np.full(2, 0.2)]
        X_max = [np.full(2, 0.9), np.full(2, 0.8)]
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2))
        cov_exp = np.array([[ 1., -0.6], [-0.6,  1.0]])
        with pytest.raises(ValueError):
            tmvbeta.fit_cov_multi_sample([X_2, X_1], X_min, X_max)
        tmvbeta.fit_cov_multi_sample([X_1, X_2], X_min, X_max)
        np.testing.assert_allclose(tmvbeta.cov, cov_exp, atol=0.01)

    def test_tmvbeta_fit_multi_sample(self) -> None:
        # Data
        X_1 = np.array([[0.2, 0.2], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]] + 12 * [[0.5, 0.5]])
        X_2 = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        X_min = [np.full(2, 0.1), np.full(2, 0.2)]
        X_max = [np.full(2, 0.9), np.full(2, 0.8)]
        a_exp = np.array([2.23535114, 2.23535063])
        b_exp = np.array([2.12188448, 2.12189103])
        cov_exp = np.array([[1.0 , -0.25407471], [-0.25407471,  1.0]])
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2))
        with pytest.raises(ValueError):
            tmvbeta.fit_multi_sample([X_2, X_1], X_min, X_max)
        tmvbeta.fit_multi_sample([X_1, X_2], X_min, X_max)
        np.testing.assert_allclose(tmvbeta.a, a_exp, atol=0.01)
        np.testing.assert_allclose(tmvbeta.b, b_exp, atol=0.01)
        np.testing.assert_allclose(tmvbeta.cov, cov_exp, atol=0.01)
    
    def test_tmvbeta_fit_multi_sample_mode(self) -> None:
        # Data
        X_1 = np.array([[0.2, 0.2], [0.9, 0.9], [0.1, 0.9], [0.9, 0.1]] + 12 * [[0.5, 0.5]])
        X_2 = np.array([[0.3, 0.3], [0.8, 0.8], [0.2, 0.8], [0.8, 0.2]] + 12 * [[0.5, 0.5]])
        X_min = [np.full(2, 0.1), np.full(2, 0.2)]
        X_max = [np.full(2, 0.9), np.full(2, 0.8)]
        mode_lb = np.full(2, 0.4)
        mode_ub = np.full(2, 0.6)
        # Test
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2))
        tmvbeta.fit_marginals_multi_sample([X_1, X_2], X_min, X_max, mode_lb=mode_lb)
        assert np.all((tmvbeta.a - 1) / (tmvbeta.a + tmvbeta.b - 2) >= mode_lb - 1e-4)
        tmvbeta = TMVBeta(np.ones(2), np.ones(2), np.eye(2))
        tmvbeta.fit_marginals_multi_sample([X_1, X_2], X_min, X_max, mode_ub=mode_ub)
        assert np.all((tmvbeta.a - 1) / (tmvbeta.a + tmvbeta.b - 2) <= mode_ub + 1e-4)
