import numpy as np


def nelson_siegel_forward(T, theta):
    return theta[0] + (theta[1] + theta[2] * T) * np.exp(-theta[3] * T)


def nelson_siegel_yield(T, theta):
    return theta[0] + (theta[1] + theta[2] / theta[3]) * (1 - np.exp(-theta[3] * T) / theta[3] / T) - theta[2] / theta[3] * np.exp(-theta[3] * T)
