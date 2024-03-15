from typing import Callable, Collection, List, Tuple, TypedDict, Union

import cvxpy as cp
import numpy as np
import numpy.typing as npt
from gridnm import GridNM
from scipy.optimize import minimize, OptimizeResult  # type: ignore
from scipy.special import beta as betafunc  # type: ignore
from scipy.special import digamma, erfcinv  # type: ignore
from scipy.stats import beta, multivariate_normal, norm, uniform  # type: ignore
from scipy.stats._multivariate import _squeeze_output  # type: ignore
from scipy.stats._distn_infrastructure import rv_continuous_frozen  # type: ignore
from sympy.functions.special.hyper import hyper

from minimax_tilting_sampler import TruncatedMVN  # type: ignore

# Custom types
anyfloat = Union[float, np.float64]
anyfloat_or_array = Union[anyfloat, npt.NDArray[np.float64]]


# Lower bound for beta distribution parameters
EPS = 1e-6


class Constraint(TypedDict):
    """
    Type dict for beta distribution mode constraints.
    """

    type: str
    fun: Callable[[npt.NDArray[np.float64]], np.float64]
    jac: Callable[[npt.NDArray[np.float64]], npt.NDArray[np.float64]]


def in_open_interval(x: anyfloat, x_min: anyfloat, x_max: anyfloat) -> bool:
    """
    Check if 'x' is a scalar in open interval (x_min, x_max).
    """
    return bool(np.isscalar(x)) and bool(x_min < x < x_max)


def in_open_cube(x: npt.NDArray[np.float64], x_min: anyfloat, x_max: anyfloat) -> np.bool_:
    """
    Check if all elements of array 'x' are in open cube (x_min, x_max) ^ x.size.
    """
    return np.all(x_min < x.flatten()) and np.all(x.flatten() < x_max)


def positive_definite(X: npt.NDArray[np.float64]) -> np.bool_:
    """
    Check if matrix 'X' is positive definite.
    """
    return np.all(np.linalg.eigvals(X) > 0)


def norm_ppf_deriv(x: anyfloat) -> np.float64:
    """
    Derivative of inverse normal CDF (PPF).
    """
    return np.sqrt(2 * np.pi) * np.exp(erfcinv(2 * x) ** 2)


def log_norm_ppf_deriv(x: anyfloat) -> np.float64:
    """
    Log of derivative of inverse normal CDF (PPF).
    """
    return 0.5 * np.log(2 * np.pi) + erfcinv(2 * x) ** 2


def betainc_deriv_a(a: anyfloat, b: anyfloat, x: anyfloat) -> anyfloat:
    """
    Derivative of incomplete beta function, I_x(a,b), with respect to 'a'.
    """
    # Catch cases in which function is constant
    if x in {0, 1}:
        return 0.0
    hyper3f2 = float(hyper((a, a, 1 - b), (1 + a, 1 + a), x))
    term1 = beta.cdf(x, a, b) * (np.log(x) - digamma(a) + digamma(a + b))
    term2 = x**a * hyper3f2 / (betafunc(a, b) * a**2)
    return term1 - term2


def betainc_grad(a: anyfloat, b: anyfloat, x: anyfloat) -> Tuple[anyfloat, anyfloat]:
    """
    Gradient of incomplete beta function, I_x(a,b), with respect to 'a' and 'b'.
    """
    return betainc_deriv_a(a, b, x), -betainc_deriv_a(b, a, 1 - x)


def array_element_lower_bound(idx: int, lb: anyfloat) -> Constraint:
    """
    Constraint dict enforcing 'x[idx] >= lb + EPS'.
    """
    jac = np.zeros(2)
    jac[idx] = 1.0
    return {"type": "ineq", "fun": lambda x: x[idx] - lb - EPS, "jac": lambda x: jac}


def beta_mode_constraint(mode: anyfloat, type: str) -> Constraint:
    """
    Constraint dict enforcing a lower or upper bound on the mode
    of a beta distribution with parameters x = [a, b].
    """
    sign = 1.0 if type == "lb" else -1.0
    return {
        "type": "ineq",
        "fun": lambda x: sign * ((1.0 - mode) * x[0] - mode * x[1] - (1.0 - 2.0 * mode)),
        "jac": lambda x: sign * np.array([1.0 - mode, -mode]),
    }


def get_beta_constraints(lb: anyfloat = 0.0, ub: anyfloat = 1.0) -> List[Constraint]:
    """
    Convert lower and upper bounds on mode to linear parameter constraints.
    """
    # Check if mode constraints bind
    lb_binds = in_open_interval(lb, 0, 1)
    ub_binds = in_open_interval(ub, 0, 1)

    if lb_binds and not ub_binds:
        # Mode lower bound
        constr_a = array_element_lower_bound(0, 1)  # a > 1
        constr_b = array_element_lower_bound(1, 0)  # b > 0
        constr_mode_lb = beta_mode_constraint(lb, "lb")
        return [constr_a, constr_b, constr_mode_lb]
    elif not lb_binds and ub_binds:
        # Mode upper bound
        constr_a = array_element_lower_bound(0, 0)  # a > 0
        constr_b = array_element_lower_bound(1, 1)  # b > 1
        constr_mode_ub = beta_mode_constraint(ub, "ub")
        return [constr_a, constr_b, constr_mode_ub]
    elif lb_binds and ub_binds:
        # Mode bracket
        constr_mode_lb = beta_mode_constraint(lb, "lb")
        constr_mode_ub = beta_mode_constraint(ub, "ub")
        return [constr_mode_lb, constr_mode_ub]

    # No mode constraint, only enforce positivity
    constr_a = array_element_lower_bound(0, 0)  # a > 0
    constr_b = array_element_lower_bound(1, 0)  # b > 0
    return [constr_a, constr_b]


def project_beta_parameters(a: anyfloat, b: anyfloat, constraints: List[Constraint]) -> Tuple[anyfloat, anyfloat, anyfloat]:
    """
    Project parameters of beta distribution to feasible region, depending on
    mode bounds. Return projected parameters and squared 2-norm of difference
    to original parameters.
    """
    # Return original parameters if they are feasible
    x0 = np.array([a, b])
    res = np.array([constr["fun"](x0) for constr in constraints])
    if np.all(res >= 0.0):
        return a, b, 0.0

    sol = minimize(
        lambda x: 0.5 * np.linalg.norm(x - x0) ** 2, x0, jac=lambda x: x - x0, constraints=constraints, method="SLSQP"
    )
    return sol.x[0], sol.x[1], sol.fun


def project_covariance(
    cov: npt.NDArray[np.float64], unit_diagonal: bool = True
) -> Tuple[npt.NDArray[np.float64], np.float64]:
    """
    Project matrix to set of positive definite matrices.
    If 'unit_diagonal' is true, require diagonal elements to be 1.
    """
    n = cov.shape[0]
    X = cp.Variable((n, n), symmetric=True)
    constraints = [X - 1e-4 * np.eye(n) >> 0]
    if unit_diagonal:
        constraints += [X[i, i] == 1.0 for i in range(n)]
    prob = cp.Problem(cp.Minimize(cp.norm(X - cov, p="fro")), constraints)
    prob.solve()
    return X.value, prob.value


def process_sample(dim: int, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Make sure input sample is 2d matrix of correct shape.
    """
    # Catch "scalar case"
    if dim == 1 and len(X.shape) == 1:
        return np.atleast_2d(X).T
    return np.atleast_2d(X)


def process_thresholds(
    dim: int,
    t_min: anyfloat,
    t_max: anyfloat,
    T_min: npt.NDArray[np.float64] | None,
    T_max: npt.NDArray[np.float64] | None,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Set and check truncation thresholds.
    """
    if T_min is None:
        T_min = np.full(dim, t_min)
    if T_max is None:
        T_max = np.full(dim, t_max)
    if T_min.size != dim or T_max.size != dim or np.any(T_min > T_max):
        raise ValueError("Invalid box specifications.")
    return T_min, T_max


def process_multi_sample_and_thresholds(
    dim: int,
    t_min: anyfloat,
    t_max: anyfloat,
    X: Collection[npt.NDArray[np.float64]],
    T_min: Collection[npt.NDArray[np.float64]] | None = None,
    T_max: Collection[npt.NDArray[np.float64]] | None = None,
) -> Tuple[
    Collection[npt.NDArray[np.float64]], Collection[npt.NDArray[np.float64]], Collection[npt.NDArray[np.float64]]
]:
    """
    Make sure input samples are 2d matrices of correct shape and
    set truncation thresholds if they are not given.
    """
    # Get number of samples
    n_samples = len(X)

    # Process samples
    X = [process_sample(dim, x) for x in X]

    # Catch no-thresholds-case
    if T_min is None:
        T_min = np.full((n_samples, dim), t_min)
    if T_max is None:
        T_max = np.full((n_samples, dim), t_max)

    # Check number of thresholds
    if not np.all(np.array([len(T_min), len(T_max)]) == n_samples):
        raise ValueError("Number of thresholds must match number of samples.")

    return X, T_min, T_max


class TBeta:
    def __init__(self, a: anyfloat, b: anyfloat, x_min: anyfloat = 0.0, x_max: anyfloat = 1.0) -> None:
        """
        Initialize truncated beta distribution with parameters 'a', 'b', 'x_min' and 'x_max'.
        """
        self.a = a
        self.b = b
        self.x_min = x_min
        self.x_max = x_max
        self.update(a, b, x_min, x_max)

    @property
    def beta(self) -> rv_continuous_frozen:
        """
        Non-truncated beta distribution with identical parameters.
        """
        return beta(self.a, self.b)

    def _in_support(self, x: anyfloat_or_array) -> npt.NDArray[np.bool_]:
        """
        Check if samples are in support of truncated distribution.
        """
        return np.logical_and(x >= self.x_min, x <= self.x_max)

    def _p_support(self) -> np.float64:
        """
        Probability of support of truncated distribution under non-truncated distribution (scaling factor).
        """
        if self.x_max == 1:
            return beta.sf(self.x_min, self.a, self.b)
        elif self.x_min == 0:
            return beta.cdf(self.x_max, self.a, self.b)
        else:
            return beta.cdf(self.x_max, self.a, self.b) - beta.cdf(self.x_min, self.a, self.b)

    def _log_p_support(self) -> np.float64:
        """
        Log-probability of support of truncated distribution under non-truncated distribution (log scaling factor).
        """
        if self.x_max == 1:
            return beta.logsf(self.x_min, self.a, self.b)
        elif self.x_min == 0:
            return beta.logcdf(self.x_max, self.a, self.b)
        else:
            return np.log(beta.cdf(self.x_max, self.a, self.b) - beta.cdf(self.x_min, self.a, self.b))

    def _loglikelihood_grad(self, mean_log_a: np.float64, mean_log_b: np.float64) -> npt.NDArray[np.float64]:
        """
        Gradient of log-likelihood with respect to a and b parameters as function of the
        sufficient statistics
        mean_log_a = np.mean(np.log(x))
        mean_log_b = np.mean(np.log(1 - x))
        """
        p_support = self._p_support()
        digamma_a_plus_b = digamma(self.a + self.b)
        betainc_grad_min_a, betainc_grad_min_b = betainc_grad(self.a, self.b, self.x_min)
        betainc_grad_max_a, betainc_grad_max_b = betainc_grad(self.a, self.b, self.x_max)
        grad_a = mean_log_a - digamma(self.a) + digamma_a_plus_b - (betainc_grad_max_a - betainc_grad_min_a) / p_support
        grad_b = mean_log_b - digamma(self.b) + digamma_a_plus_b - (betainc_grad_max_b - betainc_grad_min_b) / p_support
        return np.array([grad_a, grad_b])

    def update(
        self,
        a: anyfloat | None = None,
        b: anyfloat | None = None,
        x_min: anyfloat | None = None,
        x_max: anyfloat | None = None,
    ) -> None:
        """
        Check and update distribution parameters 'a', 'b', 'x_min' and 'x_max'
        """
        if a is not None:
            if in_open_interval(a, 0, np.inf):
                self.a = a
            else:
                raise ValueError("Parameter 'a' must be a positive.")

        if b is not None:
            if in_open_interval(b, 0, np.inf):
                self.b = b
            else:
                raise ValueError("Parameter 'b' must be a positive.")

        if x_min is not None:
            self.x_min = x_min
        if x_max is not None:
            self.x_max = x_max
        if np.any(self.x_min >= self.x_max):
            raise ValueError("'x_min' must be smaller than 'x_max'.")

    def rvs(self, size: int | None = None) -> anyfloat_or_array:
        """
        Draw sample of specified size.
        """
        U = uniform.rvs(size=size)
        return self.ppf(U)

    def pdf(self, x: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate PDF at 'x'.
        """
        x = np.atleast_1d(x)
        support = self._in_support(x)
        pdf_vec = np.zeros_like(x)
        pdf_vec[support] = beta.pdf(x[support], self.a, self.b) / self._p_support()
        return _squeeze_output(pdf_vec)

    def logpdf(self, x: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate logPDF at 'x'.
        """
        x = np.atleast_1d(x)
        support = self._in_support(x)
        logpdf_vec = np.zeros_like(x)
        logpdf_vec[support] = beta.logpdf(x[support], self.a, self.b) - self._log_p_support()
        return _squeeze_output(logpdf_vec)

    def cdf(self, x: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate CDF at 'x'.
        """
        x = np.atleast_1d(x)
        if self.x_min == 0:
            cdf_vec = beta.cdf(x, self.a, self.b) / self._p_support()
        else:
            support = x >= self.x_min
            cdf_vec = np.zeros_like(x)
            cdf_vec[support] = (
                beta.cdf(np.minimum(x[support], self.x_max), self.a, self.b) - beta.cdf(self.x_min, self.a, self.b)
            ) / self._p_support()
        return _squeeze_output(cdf_vec)

    def logcdf(self, x: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate logCDF at 'x'.
        """
        x = np.atleast_1d(x)
        if self.x_min == 0:
            logcdf_vec = beta.logcdf(x, self.a, self.b) - self._log_p_support()
        else:
            support = x >= self.x_min
            logcdf_vec = np.full_like(x, -np.inf)
            logcdf_vec[support] = (
                np.log(
                    beta.cdf(np.minimum(x[support], self.x_max), self.a, self.b) - beta.cdf(self.x_min, self.a, self.b)
                )
            ) - self._log_p_support()
        return _squeeze_output(logcdf_vec)

    def sf(self, x: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate survival function (1 - CDF) at 'x'.
        """
        x = np.atleast_1d(x)
        if self.x_max == 1:
            sf_vec = beta.sf(x, self.a, self.b) / self._p_support()
        else:
            support = x <= self.x_max
            sf_vec = np.zeros_like(x)
            sf_vec[support] = (
                beta.sf(np.maximum(x[support], self.x_min), self.a, self.b) - beta.sf(self.x_max, self.a, self.b)
            ) / self._p_support()
        return _squeeze_output(sf_vec)

    def logsf(self, x: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate log-survival function (log(1 - CDF)) at 'x'.
        """
        x = np.atleast_1d(x)
        if self.x_max == 1:
            logsf_vec = beta.logsf(x, self.a, self.b) - self._log_p_support()
        else:
            support = x <= self.x_max
            logsf_vec = -np.inf * np.ones_like(x)
            logsf_vec[support] = (
                np.log(
                    beta.sf(np.maximum(x[support], self.x_min), self.a, self.b) - beta.sf(self.x_max, self.a, self.b)
                )
                - self._log_p_support()
            )
        return logsf_vec

    def ppf(self, q: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate percent point function (inverse of CDF) at 'q'.
        """
        return beta.ppf(
            q * beta.cdf(self.x_max, self.a, self.b) + (1 - q) * beta.cdf(self.x_min, self.a, self.b),
            self.a,
            self.b,
        )

    def isf(self, q: anyfloat_or_array) -> anyfloat_or_array:
        """
        Evaluate inverse of survival function at 'q'.
        """
        return self.ppf(1 - q)

    def moment(self, order: int) -> np.float64:
        """
        Evaluate (non-centralized) moment of given order.
        """
        scale = betafunc(self.a + order, self.b) / betafunc(self.a, self.b)
        unscaled_moment = beta.cdf(self.x_max, self.a + order, self.b) - beta.cdf(self.x_min, self.a + order, self.b)
        return scale * unscaled_moment / self._p_support()

    def fit(
        self,
        x: anyfloat_or_array,
        mode_lb: anyfloat = 0.0,
        mode_ub: anyfloat = 1.0,
        method: str | None = "SLSQP",
        options: dict | None = None,
    ) -> OptimizeResult:
        """
        Fit distribution to sample vector 'x' using maximum likelihood estimation (MLE).
        Current values of 'a' and 'b' are used as initial guess.
        """
        # Make sure all samples are in support
        if not np.all(self._in_support(x)):
            raise ValueError("Cannot fit distribution. Data contains samples outside the interval [x_min, x_max].")

        # Calculate statistics
        mean_log_a = np.mean(np.log(x))
        mean_log_b = np.mean(np.log(1 - x))

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> Tuple[np.float64, npt.NDArray[np.float64]]:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            distribution parameters theta = [a, b]
            """
            self.update(*theta)
            func = np.mean(self.logpdf(x))
            grad = self._loglikelihood_grad(mean_log_a, mean_log_b)
            return -func, -grad

        constraints = get_beta_constraints(mode_lb, mode_ub)
        theta0 = np.array([self.a, self.b]).copy()
        sol = minimize(neg_loglikelihood, theta0, method=method, jac=True, constraints=constraints, options=options)
        self.update(*sol.x if sol.success else theta0)
        return sol

    def fit_multi_sample(
        self,
        X: Collection[anyfloat_or_array],
        X_min: Collection[anyfloat] | None = None,
        X_max: Collection[anyfloat] | None = None,
        mode_lb: anyfloat = 0.0,
        mode_ub: anyfloat = 1.0,
        method: str | None = "SLSQP",
        options: dict | None = None,
    ) -> OptimizeResult:
        """
        Fit distribution to heterogeneously truncated samples 'X = (x_1, ..., x_n)'
        using maximum likelihood estimation (MLE). Current parameter values are used as initial guess.
        """
        # Get number of samples
        n_samples = len(X)

        # Catch no-thresholds-case
        if X_min is None:
            X_min = n_samples * [0.0]
        if X_max is None:
            X_max = n_samples * [1.0]

        # Check number of thresholds
        if not np.all(np.array([len(X_min), len(X_max)]) == n_samples):
            raise ValueError("Number of thresholds must match number of samples.")

        # Make sure all samples are in respective support
        for x, x_min, x_max in zip(X, X_min, X_max):
            self.update(x_min=x_min, x_max=x_max)
            if not np.all(self._in_support(x)):
                raise ValueError("Cannot fit distribution. Data contains samples outside the interval [x_min, x_max].")

        # Calculate statistics and weights
        mean_log_a = [np.mean(np.log(x)) for x in X]
        mean_log_b = [np.mean(np.log(1 - x)) for x in X]
        weights = np.array([1 if isinstance(x, (float, np.float64)) else x.size for x in X], dtype=np.float64)
        weights /= np.sum(weights)

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> Tuple[np.float64, npt.NDArray[np.float64]]:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            distribution parameters theta = [a, b].
            """
            self.update(*theta)  # Update parameters
            func = np.float64(0.0)
            grad = np.zeros(2)
            for x, x_min, x_max, mla, mlb, w in zip(X, X_min, X_max, mean_log_a, mean_log_b, weights):
                self.update(x_min=x_min, x_max=x_max)  # Update truncation thresholds
                func += w * np.mean(self.logpdf(x))
                grad += w * self._loglikelihood_grad(mla, mlb)
            return -func, -grad

        constraints = get_beta_constraints(mode_lb, mode_ub)
        theta0 = np.array([self.a, self.b]).copy()
        sol = minimize(neg_loglikelihood, theta0, method=method, jac=True, constraints=constraints, options=options)
        self.update(*sol.x if sol.success else theta0)
        return sol


class TMVNormal:
    def __init__(
        self,
        mean: npt.NDArray[np.float64] | None = None,
        cov: npt.NDArray[np.float64] | anyfloat = 1.0,
        x_min: npt.NDArray[np.float64] | None = None,
        x_max: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Initialize truncated multivariate normal distribution with parameters 'mean', 'cov', 'x_min' and 'x_max'
        """
        self.mvnorm = multivariate_normal(mean, cov)
        self.x_min, self.x_max = process_thresholds(self.dim, -np.inf, np.inf, x_min, x_max)

    @property
    def dim(self) -> int:
        """
        Dimmension of distribution.
        """
        return self.mvnorm.dim

    @property
    def mean(self) -> npt.NDArray[np.float64]:
        """
        Mean of non-truncated distribution.
        """
        return self.mvnorm.mean

    @property
    def cov(self) -> npt.NDArray[np.float64]:
        """
        Covariance of non-truncated distribution.
        """
        return self.mvnorm.cov

    def _vec_to_cov(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Convert vectorized triangular matrix to covariance matrix.
        """
        cov = np.zeros((self.dim, self.dim))
        cov[np.triu_indices(self.dim)] = theta
        cov += cov.T - np.diag(cov.diagonal())
        return cov

    def _vec_to_params(self, theta: npt.NDArray[np.float64]) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert vectorized trangular matrix to covariance matrix.
        """
        mean = theta[: self.dim]
        cov = self._vec_to_cov(theta[self.dim :])
        return mean, cov

    def _params_to_vec(self) -> npt.NDArray[np.float64]:
        """
        Vectorize distribution parameters.
        """
        return np.concatenate((self.mean, self.cov[np.triu_indices(self.dim)])).copy()

    def _in_support(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """
        Check if samples are in support of truncated distribution.
        """
        return np.logical_and(
            np.all(X >= np.stack(X.shape[0] * (self.x_min,)), axis=1),
            np.all(X <= np.stack(X.shape[0] * (self.x_max,)), axis=1),
        )

    def _p_support(self) -> np.float64:
        """
        Probability of support of truncated distribution under non-truncated distribution (scaling factor).
        """
        return self.mvnorm.cdf(self.x_max, lower_limit=self.x_min)

    def _log_p_support(self) -> np.float64:
        """
        Log-probability of support of truncated distribution under non-truncated distribution (scaling factor).
        """
        return self.mvnorm.logcdf(self.x_max, lower_limit=self.x_min)

    def update(
        self,
        mean: npt.NDArray[np.float64] | None = None,
        cov: npt.NDArray[np.float64] | None = None,
        x_min: npt.NDArray[np.float64] | None = None,
        x_max: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Check and update distribution parameters 'mean', 'cov', 'x_min' and 'x_max'.
        """
        # Check and set individual parameters
        if mean is None:
            mean = self.mean
        if cov is None:
            cov = self.cov
        if x_min is None:
            x_min = self.x_min
        if x_max is None:
            x_max = self.x_max
        self.mvnorm = multivariate_normal(mean, cov)
        self.x_min, self.x_max = process_thresholds(self.dim, -np.inf, np.inf, x_min, x_max)

    def rvs(self, size: int = 1) -> npt.NDArray[np.float64]:
        """
        Draw sample of specified size.
        """
        return TruncatedMVN(self.mean, self.cov, self.x_min, self.x_max).sample(size).T

    def pdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate PDF at 'X'.
        """
        X = process_sample(self.dim, X)
        support = self._in_support(X)
        pdf_vec = np.zeros(X.shape[0])
        pdf_vec[support] = self.mvnorm.pdf(X[support]) / self._p_support()
        return _squeeze_output(pdf_vec)

    def logpdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate logPDF at 'X'.
        """
        X = process_sample(self.dim, X)
        if self._p_support() == 0:
            return _squeeze_output(-np.inf * np.ones_like(X))
        support = self._in_support(X)
        logpdf_vec = np.full(X.shape[0], -np.inf)
        logpdf_vec[support] = self.mvnorm.logpdf(X[support]) - self._log_p_support()
        return _squeeze_output(logpdf_vec)

    def cdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate CDF at 'X'.
        """
        X = process_sample(self.dim, X)
        support = np.all(X > np.stack(X.shape[0] * (self.x_min,)), axis=1)
        cdf_vec = np.zeros(X.shape[0])
        cdf_vec[support] = (
            np.array([self.mvnorm.cdf(np.minimum(self.x_max, x), lower_limit=self.x_min) for x in X[support]])
            / self._p_support()
        )
        return _squeeze_output(cdf_vec)

    def logcdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate logCDF at 'X'.
        """
        X = process_sample(self.dim, X)
        support = np.all(X > np.stack(X.shape[0] * (self.x_min,)), axis=1)
        cdf_vec = np.zeros(X.shape[0])
        cdf_vec[support] = (
            np.array([self.mvnorm.logcdf(np.minimum(self.x_max, x), lower_limit=self.x_min) for x in X[support]])
            - self._log_p_support()
        )
        return _squeeze_output(cdf_vec)

    def fit(
        self,
        X: npt.NDArray[np.float64],
        verbose: bool = False,
        method: str = "Nelder-Mead",
        options: dict = {},
        penalty_weight: anyfloat = 1e2,
    ) -> OptimizeResult:
        """
        Fit distribution to sample vector 'X' using maximum likelihood estimation (MLE).
        Current values of 'mean' and 'cov' are used as initial guess.
        """
        X = process_sample(self.dim, X)
        if verbose:
            print("Fitting truncated Gaussian distribution...")

        if not np.all(self._in_support(X)):
            raise ValueError("Cannot fit distribution. Data contains samples outside the box [x_min, x_max].")

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> anyfloat:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            distribution parameters
            """
            mean, cov = self._vec_to_params(theta)
            penalty: anyfloat = 0.0
            if not positive_definite(cov):
                cov, penalty = project_covariance(cov, unit_diagonal=False)
            self.update(mean, cov)
            val = -np.mean(self.logpdf(X)) + penalty_weight * penalty
            return np.inf if np.isnan(val) else val

        theta0 = self._params_to_vec().copy()
        sol = GridNM(neg_loglikelihood, theta0, **options).solve()
        self.update(*self._vec_to_params(sol.x if sol.success else theta0))
        return sol

    def fit_multi_sample(
        self,
        X: Collection[npt.NDArray[np.float64]],
        X_min: Collection[npt.NDArray[np.float64]] | None = None,
        X_max: Collection[npt.NDArray[np.float64]] | None = None,
        verbose: bool = False,
        options: dict = {},
        penalty_weight: anyfloat = 1e2,
    ) -> OptimizeResult:
        """
        Fit distribution to heterogeneously truncated samples 'X = (x_1, ..., x_n)'
        using maximum likelihood estimation (MLE). Current values of 'mean' and 'cov' are used as initial guess.
        """
        if verbose:
            print("Fitting parameters jointly to multiple samples...")

        # Process samples and thresholds
        X, X_min, X_max = process_multi_sample_and_thresholds(self.dim, -np.inf, np.inf, X, X_min, X_max)

        # Make sure all samples are in respective support
        for x, x_min, x_max in zip(X, X_min, X_max):
            self.update(x_min=x_min, x_max=x_max)
            if not np.all(self._in_support(x)):
                raise ValueError("Cannot fit distribution. Data contains samples outside the interval [x_min, x_max].")

        # Calculate weights
        weights = np.array([x.shape[0] for x in X], dtype=np.float64)
        weights /= np.sum(weights)

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> anyfloat:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            distribution parameters.
            """
            # Function value and penalty
            val: anyfloat = 0.0
            penalty: anyfloat = 0.0

            # Update parameters and penalty
            mean, cov = self._vec_to_params(theta)
            if not positive_definite(cov):
                cov, penalty = project_covariance(cov, unit_diagonal=False)
            self.update(mean, cov)

            for x, x_min, x_max, w in zip(X, X_min, X_max, weights):
                self.update(x_min=x_min, x_max=x_max)  # Update truncation thresholds
                val -= w * np.mean(self.logpdf(x))
            val += penalty_weight * penalty
            return np.inf if np.isnan(val) else val

        theta0 = self._params_to_vec().copy()
        sol = GridNM(neg_loglikelihood, theta0, **options).solve()
        self.update(*self._vec_to_params(sol.x if sol.success else theta0))
        return sol


class _MVBeta:
    def __init__(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64] | anyfloat = 1.0,
    ) -> None:
        """
        Initialize multivariate beta distribution with parameters 'a', 'b' and 'cov'
        """
        self.a = a
        self.b = b
        self.mvnorm = multivariate_normal(cov=cov)

    @property
    def dim(self) -> int:
        """
        Dimension of distribution.
        """
        return self.mvnorm.dim

    @property
    def cov(self) -> npt.NDArray[np.float64]:
        """
        Covariance matrix of underlying multivariate normal distribution.
        """
        return self.mvnorm.cov

    def update(
        self,
        a: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        cov: npt.NDArray[np.float64] | None = None,
    ):
        """
        Check and update distribution parameters 'a', 'b' and 'cov'.
        """
        # Update beta parameters
        if a is not None:
            if in_open_cube(a, 0, np.inf):
                self.a = a
            else:
                raise ValueError("Elements of 'a' must be positive.")
        if b is not None:
            if in_open_cube(b, 0, np.inf):
                self.b = b
            else:
                raise ValueError("Elements of 'b' must be positive.")

        # Update covariance
        if cov is not None:
            self.mvnorm = multivariate_normal(cov=cov)

        # Check that parameters are consistent
        if not (self.a.size == self.b.size == self.dim):
            raise ValueError("Inconsistent parameter dimensions.")

    def _X_to_Z(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Map MVBeta random vector to MVNormal random vector using Gaussian copula.
        """
        X = process_sample(self.dim, X)
        Z = np.zeros_like(X)
        for k in range(self.dim):
            Z[:, k] = norm.ppf(beta.cdf(X[:, k], self.a[k], self.b[k]))
        return Z

    def _Z_to_X(self, Z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Map MVNOrmal random vector to MVBeta random vector using Gaussian copula.
        """
        Z = process_sample(self.dim, Z)
        X = np.zeros_like(Z)
        norm_cdf = norm.cdf(Z)
        for k in range(self.dim):
            X[:, k] = beta.ppf(norm_cdf[:, k], self.a[k], self.b[k])
        return X

    def _vec_to_cov(self, theta: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        """
        Convert vectorized covariance matrix back to matrix.
        """
        cov = np.zeros((self.dim, self.dim))
        cov[np.triu_indices(self.dim, 1)] = theta
        cov += cov.T + np.eye(self.dim)
        return cov

    def _cov_to_vec(self) -> npt.NDArray[np.float64]:
        """
        Vectorize covariance matrix.
        """
        return self.cov[np.triu_indices(self.dim, 1)]

    def _vec_to_params(
        self, theta: npt.NDArray[np.float64]
    ) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """
        Convert jointly vectorized parameters to individual parameters.
        """
        a, b = theta[: self.dim], theta[self.dim : 2 * self.dim]
        cov = self._vec_to_cov(theta[2 * self.dim :])
        return a, b, cov

    def _params_to_vec(self) -> npt.NDArray[np.float64]:
        """
        Vectorize distribution parameters.
        """
        return np.concatenate((self.a, self.b, self._cov_to_vec()))

    def pdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate PDF at 'X'.
        """
        X = process_sample(self.dim, X)
        Z = self._X_to_Z(X)
        beta_pdf = np.array([beta.pdf(X[:, k], self.a[k], self.b[k]) for k in range(self.dim)])
        return self.mvnorm.pdf(Z) * np.prod(norm_ppf_deriv(norm.cdf(Z)), axis=1) * np.prod(beta_pdf, axis=0)

    def logpdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate logPDF at 'X'.
        """
        X = process_sample(self.dim, X)
        Z = self._X_to_Z(X)
        beta_logpdf = np.array([beta.logpdf(X[:, k], self.a[k], self.b[k]) for k in range(self.dim)])
        return self.mvnorm.logpdf(Z) + np.sum(log_norm_ppf_deriv(norm.cdf(Z)), axis=1) + np.sum(beta_logpdf, axis=0)


class TMVBeta:
    def __init__(
        self,
        a: npt.NDArray[np.float64],
        b: npt.NDArray[np.float64],
        cov: npt.NDArray[np.float64] | anyfloat = 1.0,
        x_min: npt.NDArray[np.float64] | None = None,
        x_max: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Initialize truncated multivariate beta distribution with parameters
        'a', 'b', 'cov', 'x_min' and 'x_max'.
        """
        self.mvbeta = _MVBeta(a, b, cov)
        self.x_min, self.x_max = process_thresholds(self.dim, 0.0, 1.0, x_min, x_max)

    @property
    def dim(self) -> int:
        """
        Dimension of distribution.
        """
        return self.mvbeta.dim

    @property
    def a(self) -> npt.NDArray[np.float64]:
        """
        'a' parameters of marginal beta distributions.
        """
        return self.mvbeta.a

    @property
    def b(self) -> npt.NDArray[np.float64]:
        """
        'b' parameters of marginal beta distributions.
        """
        return self.mvbeta.b

    @property
    def cov(self) -> npt.NDArray[np.float64]:
        """
        Covariance matrix of underlying (non-truncated) MVNormal distrbution.
        """
        return self.mvbeta.cov

    @property
    def tmvnorm(self) -> TMVNormal:
        """
        Underlying truncated MVNormal distrbution.
        """
        return TMVNormal(
            np.zeros(self.dim),
            self.cov,
            self.mvbeta._X_to_Z(self.x_min).flatten(),
            self.mvbeta._X_to_Z(self.x_max).flatten(),
        )

    def _in_support(self, X: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
        """
        Check if samples are in the support of truncated distribution.
        """
        return np.logical_and(
            np.all(X >= np.stack(X.shape[0] * (self.x_min,)), axis=1),
            np.all(X <= np.stack(X.shape[0] * (self.x_max,)), axis=1),
        )

    def _p_support(self) -> np.float64:
        """
        Probability of support of truncated distribution under non-truncated distribution (scaling factor).
        """
        return self.tmvnorm._p_support()

    def _log_p_support(self) -> np.float64:
        """
        Log-probability of support of truncated distribution under non-truncated distribution (log scaling factor).
        """
        return self.tmvnorm._log_p_support()

    def _project_and_update(
        self, theta: npt.NDArray[np.float64], constraints: List[List[Constraint]] | None = None
    ) -> anyfloat:
        """
        Project parameters on feasible set and update to the projected values.
        """
        penalty: anyfloat = 0.0

        # If no constraints are given, only covariance is updated
        if constraints is None:
            a, b = None, None
            cov = self.mvbeta._vec_to_cov(theta)
        else:
            a, b, cov = self.mvbeta._vec_to_params(theta)
            # Project beta parameters
            for k in range(self.dim):
                a[k], b[k], diff = project_beta_parameters(a[k], b[k], constraints[k])
                penalty += diff

        # Project covariance matrix
        if not positive_definite(cov):
            cov, diff = project_covariance(cov)
            penalty += diff

        self.update(a, b, cov)
        return penalty

    def update(
        self,
        a: npt.NDArray[np.float64] | None = None,
        b: npt.NDArray[np.float64] | None = None,
        cov: npt.NDArray[np.float64] | None = None,
        x_min: npt.NDArray[np.float64] | None = None,
        x_max: npt.NDArray[np.float64] | None = None,
    ) -> None:
        """
        Check and update distribution parameters 'a', 'b', 'cov', 'x_min' and 'x_max'.
        """
        # Update distribution parameters
        self.mvbeta.update(a, b, cov)

        # Update thresholds
        if x_min is not None or x_max is not None:
            if x_min is None:
                x_min = self.x_min
            if x_max is None:
                x_max = self.x_max
            self.x_min, self.x_max = process_thresholds(self.dim, 0.0, 1.0, x_min, x_max)

    def get_marginal(self, k: int) -> TBeta:
        """
        Get kth marginal distribution.
        """
        return TBeta(self.a[k], self.b[k], self.x_min[k], self.x_max[k])

    def rvs(self, size: int = 1) -> anyfloat_or_array:
        """
        Draw sample of given size.
        """
        Z = process_sample(self.dim, self.tmvnorm.rvs(size=size))
        X = np.zeros_like(Z)
        norm_cdf = norm.cdf(Z)
        for k in range(self.dim):
            X[:, k] = beta.ppf(norm_cdf[:, k], self.mvbeta.a[k], self.mvbeta.b[k])
        return _squeeze_output(X)

    def pdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate PDF at 'X'.
        """
        X = process_sample(self.dim, X)
        support = self._in_support(X)
        pdf_vec = np.zeros(X.shape[0])
        pdf_vec[support] = self.mvbeta.pdf(X[support]) / self._p_support()
        return _squeeze_output(pdf_vec)

    def logpdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate logPDF at 'X'.
        """
        X = process_sample(self.dim, X)
        support = self._in_support(X)
        logpdf_vec = np.full(X.shape[0], -np.inf)
        logpdf_vec[support] = self.mvbeta.logpdf(X[support]) - self._log_p_support()
        return _squeeze_output(logpdf_vec)

    def cdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate CDF at 'X'.
        """
        return self.tmvnorm.cdf(self.mvbeta._X_to_Z(X))

    def logcdf(self, X: npt.NDArray[np.float64]) -> anyfloat_or_array:
        """
        Evaluate logCDF at 'X'.
        """
        return self.tmvnorm.logcdf(self.mvbeta._X_to_Z(X))

    def fit(
        self,
        X: npt.NDArray[np.float64],
        mode_lb: npt.NDArray[np.float64] | None = None,
        mode_ub: npt.NDArray[np.float64] | None = None,
        verbose: bool = False,
        options_cov: dict = {},
        options_joint: dict = {},
    ) -> OptimizeResult:
        """
        Fit distribution to sample 'X'.

        Three MLE steps:
        1) Fit marginals
        2) Fit covariance
        3) Fit joint model
        """
        self.fit_marginals(X, mode_lb, mode_ub, verbose)
        self.fit_cov(X, verbose, options_cov)
        return self.fit_joint(X, mode_lb, mode_ub, verbose, options_joint)

    def fit_multi_sample(
        self,
        X: Collection[npt.NDArray[np.float64]],
        X_min: Collection[npt.NDArray[np.float64]] | None = None,
        X_max: Collection[npt.NDArray[np.float64]] | None = None,
        mode_lb: npt.NDArray[np.float64] | None = None,
        mode_ub: npt.NDArray[np.float64] | None = None,
        verbose: bool = False,
        options_cov: dict = {},
        options_joint: dict = {},
    ) -> OptimizeResult:
        """
        Fit distribution to heterogeneously truncated samples 'X = (x_1, ..., x_n)'.

        Three MLE steps:
        1) Fit marginals
        2) Fit covariance
        3) Fit joint model
        """
        self.fit_marginals_multi_sample(X, X_min, X_max, mode_lb, mode_ub, verbose)
        self.fit_cov_multi_sample(X, X_min, X_max, verbose, options_cov)
        return self.fit_joint_multi_sample(X, X_min, X_max, mode_lb, mode_ub, verbose, options_joint)

    def fit_marginals(
        self,
        X: npt.NDArray[np.float64],
        mode_lb: npt.NDArray[np.float64] | None = None,
        mode_ub: npt.NDArray[np.float64] | None = None,
        verbose: bool = False,
    ) -> List[OptimizeResult]:
        """
        Fit marginal distributions/copula to sample 'X'.
        """
        if verbose:
            print("Fitting marginals...")

        # Process sample and initialize parameter vectors
        X = process_sample(self.dim, X)
        a = np.zeros(self.dim)
        b = np.zeros(self.dim)

        # Check if mode constraints are set
        if mode_lb is None:
            mode_lb = np.zeros(self.dim)
        if mode_ub is None:
            mode_ub = np.ones(self.dim)

        # Fit marginals
        sols = []
        tbeta = TBeta(1, 1)
        for k in range(self.dim):
            tbeta.update(self.a[k], self.b[k], self.x_min[k], self.x_max[k])
            sols.append(tbeta.fit(X[:, k], mode_lb=mode_lb[k], mode_ub=mode_ub[k]))
            a[k], b[k] = tbeta.a, tbeta.b

        if all([sol.success for sol in sols]):
            self.update(a, b)
        return sols

    def fit_marginals_multi_sample(
        self,
        X: Collection[npt.NDArray[np.float64]],
        X_min: Collection[npt.NDArray[np.float64]] | None = None,
        X_max: Collection[npt.NDArray[np.float64]] | None = None,
        mode_lb: npt.NDArray[np.float64] | None = None,
        mode_ub: npt.NDArray[np.float64] | None = None,
        verbose: bool = False,
    ) -> List[OptimizeResult]:
        """
        Fit marginal distributions/copula to heterogeneously truncated samples 'X = (x_1, ..., x_n)'.
        """
        if verbose:
            print("Fitting marginals to multiple samples...")

        # Process samples and thresholds
        X, X_min, X_max = process_multi_sample_and_thresholds(self.dim, 0.0, 1.0, X, X_min, X_max)

        # Initialize parameter vectors
        a = np.zeros(self.dim)
        b = np.zeros(self.dim)

        # Check if mode constraints are set
        if mode_lb is None:
            mode_lb = np.zeros(self.dim)
        if mode_ub is None:
            mode_ub = np.ones(self.dim)

        # Fit marginals
        sols = []
        tbeta = TBeta(1, 1)
        for k in range(self.dim):
            tbeta.update(self.a[k], self.b[k])
            X_k = [x[:, k] for x in X]
            X_min_k = [x_min[k] for x_min in X_min]
            X_max_k = [x_max[k] for x_max in X_max]
            sols.append(
                tbeta.fit_multi_sample(X_k, X_min=X_min_k, X_max=X_max_k, mode_lb=mode_lb[k], mode_ub=mode_ub[k])
            )
            a[k], b[k] = tbeta.a, tbeta.b

        if all([sol.success for sol in sols]):
            self.update(a, b)
        return sols

    def fit_cov(
        self, X: npt.NDArray[np.float64], verbose=False, options: dict = {}, penalty_weight: anyfloat = 1e2
    ) -> OptimizeResult:
        """
        Fit underlying truncated normal distribution to sample matrix 'X'
        using maximum likelihood estimation (MLE). Current covariance matrix is used for initial guess.
        """
        if verbose:
            print("Fitting covariance matrix...")

        # Make sure all samples lie in support
        X = process_sample(self.dim, X)
        if not np.all(self._in_support(X)):
            raise ValueError("Cannot fit distribution. Data contains samples outside the box [x_min, x_max].")

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> anyfloat:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            covariance matrix theta.
            """
            penalty = self._project_and_update(theta)
            val = -np.mean(self.logpdf(X)) + penalty_weight * penalty
            return np.inf if np.isnan(val) else val

        theta0 = self.mvbeta._cov_to_vec().copy()
        sol = GridNM(neg_loglikelihood, theta0, **options).solve()
        if sol.success:
            self._project_and_update(sol.x)
        else:
            self.update(cov=self.mvbeta._vec_to_cov(theta0))
        return sol

    def fit_cov_multi_sample(
        self,
        X: Collection[npt.NDArray[np.float64]],
        X_min: Collection[npt.NDArray[np.float64]] | None = None,
        X_max: Collection[npt.NDArray[np.float64]] | None = None,
        verbose=False,
        options: dict = {},
        penalty_weight: anyfloat = 1e2,
    ) -> OptimizeResult:
        """
        Fit underlying truncated normal distribution to heterogeneously truncated samples 'X = (x_1, ..., x_n)'
        using maximum likelihood estimation (MLE). Current covariance matrix is used for initial guess.
        """
        if verbose:
            print("Fitting covariance matrix to multiple samples...")

        # Process samples and thresholds
        X, X_min, X_max = process_multi_sample_and_thresholds(self.dim, 0.0, 1.0, X, X_min, X_max)

        # Make sure all samples are in respective support
        for x, x_min, x_max in zip(X, X_min, X_max):
            self.update(x_min=x_min, x_max=x_max)
            if not np.all(self._in_support(x)):
                raise ValueError("Cannot fit distribution. Data contains samples outside the interval [x_min, x_max].")

        # Calculate weights
        weights = np.array([x.shape[0] for x in X], dtype=np.float64)
        weights /= np.sum(weights)

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> anyfloat:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            covariance matrix theta.
            """
            penalty = self._project_and_update(theta)
            val: anyfloat = 0.0
            for x, x_min, x_max, w in zip(X, X_min, X_max, weights):
                self.update(x_min=x_min, x_max=x_max)  # Update truncation thresholds
                val -= w * np.mean(self.logpdf(x))
            val += penalty_weight * penalty
            return np.inf if np.isnan(val) else val

        theta0 = self.mvbeta._cov_to_vec().copy()
        sol = GridNM(neg_loglikelihood, theta0, **options).solve()
        if sol.success:
            self._project_and_update(sol.x)
        else:
            self.update(cov=self.mvbeta._vec_to_cov(theta0))
        return sol

    def fit_joint(
        self,
        X: npt.NDArray[np.float64],
        mode_lb: npt.NDArray[np.float64] | None = None,
        mode_ub: npt.NDArray[np.float64] | None = None,
        verbose: bool = False,
        options: dict = {},
        penalty_weight: anyfloat = 1e2,
    ) -> OptimizeResult:
        """
        Fit distribution to sample matrix 'X' using maximum likelihood estimation (MLE).
        Marginal parameters and covariance matrix are optimized jointly. Current values
        of 'a', 'b' and 'cov' are used for initial guess.
        """
        if verbose:
            print("Fitting parameters jointly...")

        # Make sure all samples lie in support
        X = process_sample(self.dim, X)
        if not np.all(self._in_support(X)):
            raise ValueError("Cannot fit distribution. Data contains samples outside the box [x_min, x_max].")

        # Check if mode constraints are set
        if mode_lb is None:
            mode_lb = np.zeros(self.dim)
        if mode_ub is None:
            mode_ub = np.ones(self.dim)

        # Get constrains
        constraints = [get_beta_constraints(lb, ub) for (lb, ub) in zip(mode_lb, mode_ub)]

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> anyfloat:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            distribution parameters.
            """
            penalty = self._project_and_update(theta, constraints)
            val = -np.mean(self.logpdf(X)) + penalty_weight * penalty
            return np.inf if np.isnan(val) else val

        theta0 = self.mvbeta._params_to_vec().copy()
        sol = GridNM(neg_loglikelihood, theta0, **options).solve()
        if sol.success:
            self._project_and_update(sol.x, constraints)
        else:
            self.update(*self.mvbeta._vec_to_params(theta0))
        return sol

    def fit_joint_multi_sample(
        self,
        X: Collection[npt.NDArray[np.float64]],
        X_min: Collection[npt.NDArray[np.float64]] | None = None,
        X_max: Collection[npt.NDArray[np.float64]] | None = None,
        mode_lb: npt.NDArray[np.float64] | None = None,
        mode_ub: npt.NDArray[np.float64] | None = None,
        verbose: bool = False,
        options: dict = {},
        penalty_weight: anyfloat = 1e2,
    ) -> OptimizeResult:
        """
        Fit distribution to heterogeneously truncated samples 'X = (x_1, ..., x_n)'
        using maximum likelihood estimation (MLE). Current values of 'a', 'b' and 'cov' are used for initial guess.
        """
        if verbose:
            print("Fitting parameters jointly to multiple samples...")

        # Process samples and thresholds
        X, X_min, X_max = process_multi_sample_and_thresholds(self.dim, 0.0, 1.0, X, X_min, X_max)

        # Make sure all samples are in respective support
        for x, x_min, x_max in zip(X, X_min, X_max):
            self.update(x_min=x_min, x_max=x_max)
            if not np.all(self._in_support(x)):
                raise ValueError("Cannot fit distribution. Data contains samples outside the interval [x_min, x_max].")

        # Calculate weights
        weights = np.array([x.shape[0] for x in X], dtype=np.float64)
        weights /= np.sum(weights)

        # Check if mode constraints are set
        if mode_lb is None:
            mode_lb = np.zeros(self.dim)
        if mode_ub is None:
            mode_ub = np.ones(self.dim)

        # Get constrains
        constraints = [get_beta_constraints(lb, ub) for (lb, ub) in zip(mode_lb, mode_ub)]

        def neg_loglikelihood(theta: npt.NDArray[np.float64]) -> anyfloat:
            """
            MLE objective/negative log-likelihood as a function of vectorized
            distribution parameters.
            """
            penalty = self._project_and_update(theta, constraints)
            val: anyfloat = 0.0
            for x, x_min, x_max, w in zip(X, X_min, X_max, weights):
                self.update(x_min=x_min, x_max=x_max)  # Update truncation thresholds
                val -= w * np.mean(self.logpdf(x))
            val += penalty_weight * penalty
            return np.inf if np.isnan(val) else val

        theta0 = self.mvbeta._params_to_vec().copy()
        sol = GridNM(neg_loglikelihood, theta0, **options).solve()
        if sol.success:
            self._project_and_update(sol.x, constraints)
        else:
            self.update(*self.mvbeta._vec_to_params(theta0))
        return sol
