import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma, beta
import statsmodels.api as sm


def aic(fit):
    return 2 * fit.fun + 2 * len(fit.x)


def bic(fit, n):
    return 2 * fit.fun + np.log(n) * len(fit.x)


def reduce_model(x, y, fit_f):
    """A simplistic version of the stepAIC function from R's MASS package"""
    while len(x.columns) > 1:
        baseline_fit = fit_f(sm.add_constant(x), y)
        print(f'Variables: {list(x.columns)}')
        
        res = []
        res.append(baseline_fit.aic)

        for col in x.columns:
            reduced_x = x.drop(col, axis=1)
            fit = fit_f(sm.add_constant(reduced_x), y)
            res.append(fit.aic)

        res = pd.DataFrame(res, columns=['AIC'], index=['<none>'] + list(x.columns))
        res.sort_values(by='AIC', inplace=True)
        print(res)
        
        if res.iloc[0, 0] < baseline_fit.aic:
            print(f'Dropping variable: {res.index[0]}\n')
            x = x.drop(res.index[0], axis=1)
        else:
            print('Stopping')
            break
    print('One variable left in the model - stopping')
    return list(x.columns)
