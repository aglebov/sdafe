import numpy as np
import pandas as pd
import pytest

from cov_trob import cov_trob


@pytest.fixture(scope="session")
def berndt_data():
    return pd.read_csv('data/berndtInvest.csv', index_col=0, parse_dates=[0])


def test_cov_trob(berndt_data):
    ret = berndt_data.iloc[:, :4]

    res = cov_trob(ret.values, nu=4, cor=True)

    assert res['n.obs'] == 120
    assert res['iter'] == 4

    center_expected = np.array([0.015843852, 0.017612774, -0.006237575, 0.013627204])
    np.testing.assert_array_almost_equal(res['center'], center_expected, decimal=9)

    cov_expected = np.array([
        [0.004145915, 0.0010581124, 0.0027909589, 0.0031253951],
        [0.001058112, 0.0017586623, 0.0005372638, 0.0005138348],
        [0.002790959, 0.0005372638, 0.0070923425, 0.0027754949],
        [0.003125395, 0.0005138348, 0.0027754949, 0.0104424587],
    ])
    np.testing.assert_array_almost_equal(res['cov'], cov_expected, decimal=9)

    cor_expected = np.array([
        [1.0000000, 0.3918595, 0.5146931, 0.4749997],
        [0.3918595, 1.0000000, 0.1521253, 0.1199032],
        [0.5146931, 0.1521253, 1.0000000, 0.3225107],
        [0.4749997, 0.1199032, 0.3225107, 1.0000000],
    ])
    np.testing.assert_array_almost_equal(res['cor'], cor_expected, decimal=7)
