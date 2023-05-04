import numpy as np
import scipy.stats as stats
from scipy.special import gamma, beta


def dged(x, loc=0, scale=1, nu=2):
    """Generalided error distribution, as defined in section 5.7 of SDAFE"""
    lnu = np.sqrt(2 ** (-2 / nu) * gamma(1 / nu) / gamma(3 / nu))
    k = nu / lnu / 2 ** (1 + 1 / nu) / gamma(1 / nu)
    return k * np.exp(-1/2 * np.abs((x - loc) / scale / lnu) ** nu) / scale


def _dsged(x, nu, xi):
    """Skewed generalised error distribution (from https://rdrr.io/cran/fGarch/src/R/dist-sged.R)"""
    lnu = np.sqrt(2 ** (-2 / nu) * gamma(1 / nu) / gamma(3 / nu))
    
    m1 = 2 ** (1 / nu) * lnu * gamma(2 / nu) / gamma(1 / nu)
    mu = m1 * (xi - 1 / xi)
    sigma = np.sqrt((1 - m1 ** 2) * (xi ** 2 + 1 / xi ** 2) + 2 * m1 ** 2 - 1)
    
    z = x * sigma + mu
    Xi = xi ** np.sign(z)
    g = 2 / (xi + 1 / xi)
    Density = g * dged(z / Xi, nu=nu)
    
    return Density * sigma


def dsged(x, loc=0, scale=1, nu=2, xi=1):
    return _dsged((x - loc) / scale, nu=nu, xi=xi) / scale


def _sstd_params(nu, xi):
    m1 = 2 * np.sqrt(nu - 2) / (nu - 1) / beta(0.5, nu / 2)
    mu = m1 * (xi - 1 / xi)
    sigma = np.sqrt((1 - m1 ** 2) * (xi ** 2 + 1 / xi ** 2) + 2 * m1 ** 2 - 1)
    g = 2 / (xi + 1 / xi)
    return mu, sigma, g


def _dsstd(x, nu, xi):
    """Implementation of skewed Student's t-distribution from https://github.com/cran/fGarch/blob/master/R/dist-sstd.R"""
    mu, sigma, g = _sstd_params(nu, xi)
    
    z = x * sigma + mu
    Xi = xi ** np.sign(z)
    Density = g * stats.t.pdf(z / Xi, scale=np.sqrt((nu - 2) / nu), df=nu)

    return Density * sigma


def dsstd(x, mean, sd, nu, xi):
    """Skewed Student's t-distribution"""
    return _dsstd(x=(x - mean) / sd, nu=nu, xi=xi) / sd


def qsstd(p, mean, sd, nu, xi):
    """Implementation of skewed Student's t-distribution from https://github.com/cran/fGarch/blob/master/R/dist-sstd.R"""
    mu, sigma, g = _sstd_params(nu, xi)

    pxi = p - (1 / (1 + xi ** 2)) # not p - 1/2
    sig = np.sign(pxi)  # not p - 1/2
    Xi = xi ** sig
    p = (np.heaviside(pxi, 0.5) - sig * p) / (g * Xi)  # pxi, not p - 1/2

    # the quantile of the standardised skewed t-distribution
    q = (-sig * stats.t.ppf(p, scale=Xi * np.sqrt((nu - 2) / nu), df=nu) - mu) / sigma

    return q * sd + mean


def rsstd(n, mean, sd, nu, xi):
    """Implementation of skewed Student's t-distribution from https://github.com/cran/fGarch/blob/master/R/dist-sstd.R"""
    # Generate Random Deviates:
    weight = xi / (xi + 1 / xi)
    z = stats.uniform.rvs(size=n, loc=-weight, scale=1)
    Xi = xi ** np.sign(z)
    r = -np.abs(stats.t.rvs(size=n, scale=np.sqrt((nu - 2) / nu), df=nu)) / Xi * np.sign(z)

    # Scale:
    mu, sigma, _ = _sstd_params(nu, xi)
    return (r - mu) / sigma * sd + mean
