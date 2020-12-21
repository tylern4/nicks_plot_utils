import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from lmfit import Parameters, minimize, report_fit
import functools
import operator

__ALPHA__ = 0.8


class Hist2D(bh.Histogram):
    def __init__(self,
                 x_left: float = -1,
                 x_right: float = 1,
                 x_bins: float = 100,
                 x_name: str = None,
                 y_left: float = -1,
                 y_right: float = 1,
                 y_bins: float = 100,
                 y_name: str = None,
                 *args, **kwargs) -> None:
        super(Hist2D, self).__init__(bh.axis.Regular(x_bins, x_left, x_right, metadata=x_name),
                                     bh.axis.Regular(
                                         y_bins, y_left, y_right, metadata=x_name),
                                     *args, **kwargs)
        self.color = None
        self.x_name = x_name
        self.x_left = x_left
        self.x_right = x_right
        self.xs = np.linspace(x_left, x_right, 500)

        self.y_name = y_name
        self.y_left = y_left
        self.y_right = y_right
        self.ys = np.linspace(y_left, y_right, 500)

    def plot(self, ax=None,
             filled: bool = False, alpha: float = __ALPHA__,
             cmap=None, density: bool = True,  colorbar: bool = True, zeros: bool = True):
        if not ax:
            ax = plt.gca()
        if density:
            # Compute the areas of each bin
            areas = functools.reduce(operator.mul, self.axes.widths)
            # Compute the density
            zvalues = self.view() / self.sum() / areas
        else:
            zvalues = self.view()

        zvalues = zvalues if zeros else np.where(zvalues == 0, np.nan,
                                                 zvalues)

        pc = ax.pcolormesh(*self.axes.edges.T, zvalues.T,
                           cmap=cmap if cmap else 'viridis')

        ax.set_xlabel(self.axes[0].metadata)
        ax.set_ylabel(self.axes[1].metadata)
        if colorbar:
            plt.gcf().colorbar(pc, ax=ax, aspect=30)
        return pc

    def fill(self, data_x, data_y):
        # If we pass in a pandas series then rename the axes to the series name
        # Cool trick to not have to add labels to the histograms
        if isinstance(data_x, pd.Series):
            self.axes[0].metadata = data_x.name
        if isinstance(data_y, pd.Series):
            self.axes[1].metadata = data_y.name

        return super(Hist2D, self).fill(data_x, data_y)

    def _gaussian2D(self, x, y, cen_x, cen_y, sig_x, sig_y, offset):
        func = np.exp(-(((cen_x-x)/sig_x)**2 + ((cen_y-y)/sig_y)**2)/2.0)
        return func + offset

    def gaussian2D(self, x, y, height, cen_x, cen_y, sig_x, sig_y, offset):
        func = np.exp(-(((cen_x-x)/sig_x)**2 +
                        ((cen_y-y)/sig_y)**2)/2.0)
        return (height * (func + offset)).ravel()

    def _residuals(self, p, x, y, z):
        height = p["height"].value
        cen_x = p["cen_x"].value
        cen_y = p["cen_y"].value
        sigma_x = p["sigma_x"].value
        sigma_y = p["sigma_y"].value
        offset = p["offset"].value
        return (z - height*self._gaussian2D(x, y, cen_x, cen_y, sigma_x, sigma_y, offset))

    def fitGausian(self, ax=None, label: bool = False):
        if not ax:
            ax = plt.gca()
        initial = Parameters()
        initial.add("height", value=1.)
        initial.add("cen_x", value=0.)
        initial.add("cen_y", value=0.)
        initial.add("sigma_x", value=1.)
        initial.add("sigma_y", value=3.)
        initial.add("offset", value=0.)

        x, y = self.axes.edges.T
        areas = functools.reduce(operator.mul, self.axes.widths)
        g = self.view() / self.sum() / areas
        g = g.T
        fit = minimize(self._residuals, initial, args=(x[:, 1:], y[1:, :], g))

        X, Y = np.meshgrid(x[:, 1:], y[1:, :])
        Z = self.gaussian2D(X, Y, fit.params['height'].value,
                            fit.params['cen_x'].value,
                            fit.params['cen_y'].value,
                            fit.params['sigma_x'].value,
                            fit.params['sigma_y'].value,
                            fit.params['offset'].value).reshape(X.shape[0], Y.shape[0])
        CS = ax.contour(X, Y, Z, colors='w', linestyles='-')
        if label:
            CS.levels = [f'{s:.2f}' for s in CS.levels]
            ax.clabel(CS, CS.levels, inline=True, fmt=r'%r \%%', fontsize=10)

        return fit
