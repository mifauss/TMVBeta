TMVBeta
=====================================

Truncated Multivariate Beta Distribution on Unit Hypercube

## Description

Python implementation of the following distributions
- Truncated beta distribution
- Truncated multivariate normal distribution
- Truncated multivariate beta distribution on unit hypercube

All distributions support common methods, such as, `pdf`, `cdf`, `logpdf`, etc. In addition, convenient methods are provided to
- efficiently draw samples
- estimate parameters using maximum likelihood based on homogeneously or heterogeneously truncated samples

Maximum likelihood estimation is performed using a grid-restrained version of the Nelder-Mead algorithm, which is guarantee to converge to a stationary point. See the documentation in `src/tmvbeta/tmvbeta.py` for details.


## Implementation Details 

There is no unique definition of the multivariate beta distribution on the hypercube. The distribution implemented here is generated from a standard normal distribution via a Gaussian copula:
```math
\boldsymbol{X} \sim F_{\boldsymbol{\alpha}, \boldsymbol{\beta}}^{-1} \bigl( \Phi(\boldsymbol{Z}) \bigr), \quad \text{with}\quad \boldsymbol{Z} \sim \mathcal{N}(\boldsymbol{0}, \boldsymbol{\Sigma}),
```
where $F_{\alpha, \beta}$ and $\Phi$ denote the CDF of the beta distribution and the standard normal distribution, respectively, and the covariance matrix of $\boldsymbol{Z}$ has ones on its diagonal, $[\boldsymbol{\Sigma}]_{ii} = 1$


## Getting Started

Three examples of how to use the distributions in this library are given below. For more details see the documentation in `src/tmvbeta/tmvbeta.py`.

1. Sample from beta distribution truncated to interval $[0.2, 0.75]$, and estimate parameters from the sample:
```python
from tmvbeta import TBeta

# parameters
a, b = 2, 4
x_min, x_max = 0.2, 0.75

# initialize distribution
tbeta = TBeta(a, b, x_min, x_max)

# generate sample
x = tbeta.rvs(size=100)

# estimate parameters
tbeta.fit(x)

# print estimated parameters
print(tbeta.a, tbeta.b)
```

2. Sample from 2D standard normal distribution truncated to rectangle $[-1, 2]^2$, and estimate parameters from the sample:
```python
import numpy as np
from tmvbeta import TMVNormal

# parameters
mean = np.zeros(2)
cov = np.eye(2)
x_min = np.full(-1, 2)
x_max = np.full(2, 2)

# initialize distribution
tmvnorm = TMVNormal(a, b, x_min, x_max)

# generate sample
x = tmvnorm.rvs(size=100)

# estimate parameters
tmvnorm.fit(x)

# print estimated parameters
print(tmvnorm.mean, tmvnorm.cov)
```

3. Sample from 2D beta distribution on hypercube truncated to rectangle $[0.1, 0.9]^2$, and estimate parameters from the sample:
```python
import numpy as np
from tmvbeta import TMVBeta

# parameters
a = np.full(2, 2)
b = np.full(4, 2)
cov = np.eye(2)
x_min = np.full(0.1, 2)
x_max = np.full(0.9, 2)

# initialize distribution
tmvbeta = TMVBeta(a, b, cov, x_min, x_max)

# generate sample
x = tmvbeta.rvs(size=100)

# estimate parameters
tmvbeta.fit(x)

# print estimated parameters
print(tmvbeta.a, tmvbeta.b, tmvbeta.cov)
```