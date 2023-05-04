import warnings
from typing import Any, Optional, Union, Dict

import numpy as np


def cov_trob(x: np.array, wt: Optional[np.array] = None, cor: Optional[bool] = False,
             center: Union[bool, np.array] = True, nu: float = 5, maxit: int = 25, tol: float = 0.01) -> Dict[str, Any]:
    """Fits the parameters of a t-distribution to the provided data

    This implementation is a translation of the cov.trob function implementation in the MASS package in R
    (https://github.com/cran/MASS/blob/master/R/cov.trob.R) by Brian Ripley.
    The original R implementation is based on the algorithm in J. T. Kent, D. E. Tyler and Y. Vardi (1994)
    A curious likelihood identity for the multivariate t-distribution. Communications in Statistics—Simulation
    and Computation 23, 441–453.

    Parameters
    ----------
    x: np.array
        An array of observations, with rows representing individual observations, and columns the observed variables
    wt: Optional[np.array]
        The weights of the observations (rows) in x. By defaults, all observations are taken to have equal weight.
    cor: Optional[bool]
        Whether to return the correlation matrix calculated from the covariance matrix. False by default.
    center: Union[bool, np.array]
        If a boolean value is provided, indicates whether to estimate the location parameter for each variable (True)
        or assume that the location is zero for each. If a number vector is provided, use it as the location and do not
        estimate it.
    nu: float
        The number of degrees of freedom of the fitted t-distribution. Default: 5
    maxit: int
        Limits the number of iterations that the algorithm performs. Default: 25
    tol: float
        The tolerance used in the convergence criterion of the algorithm. Default: 0.01

    Returns
    -------
    """
    assert maxit > 0, 'maxit must be greater than 0'
    assert tol > 0, 'tol must be greater than 0'

    def test_values(x):
        if np.any(np.isnan(x) | np.isinf(x)):
            raise ValueError('missing or infinite values in x')

    n, p = x.shape
    test_values(x)

    if miss_wt := wt is None:
        wt = np.repeat(1, x.shape[0])
    else:
        wt0 = wt
        test_values(wt)
        if len(wt) != n:
            raise ValueError('length of wt must equal number of observations')
        if np.any(wt < 0):
            raise ValueError('negative weights not allowed')
        if np.sum(wt) == 0.0:
            raise ValueError('no positive weights')

        # drop rows with zero weight
        x = x.loc[wt > 0]
        wt = wt[wt > 0]
        n = x.shape[0]

    def diag_rect(v):
        t = np.zeros((min(n, p), p))
        np.fill_diagonal(t, v)
        return t

    wt = wt.reshape(-1, 1)  # make sure the weights are a column-vector

    loc = np.sum(x * wt, axis=0) / np.sum(wt)  # the initial estimate of the location parameters for each variable

    if type(center) is np.array:
        if len(center) != p:
            raise ValueError('center is not the right length')
        loc = center
    elif type(center) is bool and not center:
        loc = np.zeros(p)

    use_loc = type(center) is bool and center

    w = wt * (1 + p / nu)  # the starting weights
    endit = 0  # will contain the number of iterations taken to converge
    for i in range(maxit):
        w0 = w
        X = x - loc
        _, s, vh = np.linalg.svd(X * np.sqrt(w / np.sum(w)))
        wX = X @ vh.T @ diag_rect(1 / s)
        Q = (wX ** 2 @ np.ones(p)).reshape(-1, 1)  # calculate the distances
        w = wt * (nu + p) / (nu + Q)  # recalculate the weights

        if use_loc:
            loc = np.sum(x * w, axis=0) / np.sum(w)

        # the convergence criterion: the changes in weights are below the chosen tolerance threshold
        if np.all(np.abs(w - w0) < tol):
            break

        endit = i + 1

    if endit == maxit or np.abs(np.mean(w) - np.mean(wt)) > tol or np.abs(np.mean(Q * w) / p - 1) > tol:
        warnings.warn('Probable convergence failure')

    Xw = X * np.sqrt(w)
    cov = Xw.T @ Xw / np.sum(wt)

    if miss_wt:
        ans = {'cov': cov, 'center': loc, 'n.obs': n}
    else:
        ans = {'cov': cov, 'center': loc, 'wt': wt0, 'n.obs': n}

    if cor:
        sd_inv = np.diag(1 / np.sqrt(np.diag(cov)))
        cor = sd_inv @ cov @ sd_inv
        ans['cor'] = cor

    ans['iter'] = endit

    return ans



