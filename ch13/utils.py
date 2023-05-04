import numpy as np
import scipy.stats as stats

import pmdarima as pm


def predict(fit, v, n):
    """Forecast values based on a VAR model fit"""
    assert n > 0
    p = fit['order'][0]
    mu = fit['x.mean'].reshape(-1, 1)
    phis = fit['ar']
    
    res = None
    for i in range(n):
        Ys = np.expand_dims(np.flipud(v[-p:, :]), 2)
        res1 = np.squeeze(mu + np.sum(phis @ (Ys - mu), axis=0))
        res = np.vstack([res, res1]) if res is not None else res1
        v = np.vstack([v, res1])
    return res


def refit(fit0, x, niter=10, print_models=True, rng=None, burnin=20):
    """Check how often auto_arima selects the same AR model"""
    rng = np.random.default_rng(1998852) if rng is None else rng
    n = x.shape[0]
    phis = fit0.arparams()
    p = len(phis)
    same_selected = 0
    for i in range(niter):
        eps = stats.norm.rvs(size=n + burnin, scale=np.sqrt(fit0.params()['sigma2']), random_state=rng)
        y = np.zeros(n + burnin)
        for t in range(p, n + burnin):
            y[t] = np.sum(y[t-p:t] * phis) + eps[t]
        y = y[burnin:]
        y = np.cumsum(y)

        fit = pm.auto_arima(y, information_criterion='bic')
        if fit.order == fit0.order and fit.seasonal_order == fit0.seasonal_order:
            same_selected += 1
        if print_models:
            print(fit)
    return same_selected / niter