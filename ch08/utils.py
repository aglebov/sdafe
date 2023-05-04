import numpy as np

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri, numpy2ri, default_converter

np_cv_rules = default_converter + numpy2ri.converter

copula = importr('copula')


def fit_copula(copula_def, start, data):
    with np_cv_rules.context():
        if start is not None:
            C = copula.fitCopula(copula=copula_def, data=data, method='ml', start=robjects.FloatVector(start))
        else:
            C = copula.fitCopula(copula=copula_def, data=data, method='ml')
        est = C.slots['estimate']
        aic = -2 * copula.loglikCopula(param=robjects.FloatVector(est), u=data, copula=copula_def)[0] + 2 * len(est)
        return est, aic

    
def plot_empirical_copula(ax, xs, ys, zs, levels):
    n = len(xs)
    cs = ax.contour(xs, xs, zs, levels=levels, colors='red', linewidths=0.5);
    ax.clabel(cs, cs.levels, inline=True, fontsize=6);

    
def plot_copula(ax, copula_def, xs, levels, value_fun):
    n = len(xs)
    grid = np.vstack([np.repeat(xs, n), np.tile(xs, n)]).T
    with np_cv_rules.context():
        zs = value_fun(grid, copula_def)
    cs = ax.contour(xs, xs, zs.reshape((n,n)).T, levels=levels, colors='black', linewidths=0.5);
    ax.clabel(cs, cs.levels, inline=True, fontsize=6);
