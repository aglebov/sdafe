import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf


def plot_residuals(res, res_std=None, zero=True):
    rows = 2 if res_std is not None else 1
    fig, axs = plt.subplots(rows, 3, figsize=(15, rows * 4 + (rows - 1)))

    if rows > 1:
        row = axs[0]
    else:
        row = axs
        
    row[0].plot(res);
    row[0].set_title('Residuals');

    plot_acf(res, ax=row[1], zero=zero, auto_ylims=not zero);
    row[1].set_title('ACF of residuals');

    plot_acf(res ** 2, ax=row[2], zero=zero, auto_ylims=not zero);
    row[2].set_title('ACF of squared residuals');

    if res_std is not None:
        row = axs[1]

        row[0].plot(res_std);
        row[0].set_title('Standardised residuals');

        plot_acf(res_std, ax=row[1], zero=zero, auto_ylims=not zero);
        row[1].set_title('ACF of standardised residuals');

        plot_acf(res_std ** 2, ax=row[2], zero=zero, auto_ylims=not zero);
        row[2].set_title('ACF of squared standardised residuals');
