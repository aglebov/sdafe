import numpy as np
import pandas as pd
import scipy.stats as stats
from scipy.special import gamma, beta
import statsmodels.api as sm


def aic(fit):
    return 2 * fit.fun + 2 * len(fit.x)


def bic(fit, n):
    return 2 * fit.fun + np.log(n) * len(fit.x)
