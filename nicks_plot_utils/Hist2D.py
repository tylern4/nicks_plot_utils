from typing import List
import boost_histogram as bh
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import pandas as pd
from lmfit import Parameters, minimize
from lmfit.models import GaussianModel
import functools
import operator
import warnings
from .Hist1D import Hist1D
from .Scatter import Scatter

__ALPHA__ = 0.8


class Hist2D:
    """Hist2D is a wrapper around a boost histogram which sets up a 2D histogram and adds simples plots and fitting.

    Args:
        [Required]
            _bins -> int : Sets the number of bins to use
        [Semi-Required]
            _range -> List [Required unless data is suplied)] Sets the xrange of the histogram
        [Optional]
            _data -> Array like : Data to fill the histogram with
            _name -> string : Name to be used with automatic axes labes

        *args, **kwargs are passed to bh.Histogram
    """

    def __init__(self,
                 xdata=None,
                 ydata=None,
                 xbins: float = 100,
                 xrange: List = None,
                 xname: str = None,
                 ybins: float = 100,
                 yrange: List = None,
                 yname: str = None,
                 boost_hist=None,
                 *args, **kwargs) -> None:

        self.xname = xname
        self.yname = yname

        if boost_hist is not None:
            self.hist = boost_hist
            return
        self.xbins = xbins

        # Get the left and right bin edges either from range or dataset with fallbacks
        if xdata is not None:
            self.xleft = np.min(xdata)
            self.xright = np.max(xdata)
        if xrange is not None:
            if isinstance(xrange, list):
                self.xleft = xrange[0]
                self.xright = xrange[1]
            elif isinstance(xrange, float):
                self.xleft = -1*np.abs(xrange)
                self.xright = np.abs(xrange)
        try:
            if self.xleft is None or self.xright is None:
                self.xleft = -1.0
                self.xright = 1.0
        except AttributeError:
            self.xleft = -1.0
            self.xright = 1.0
            print("Need to start with data or set xrange=[left,right]")

        self.ybins = ybins

        # Get the left and right bin edges either from range or dataset with fallbacks
        if ydata is not None:
            self.yleft = np.min(ydata)
            self.yright = np.max(ydata)
        if xrange is not None:
            if isinstance(yrange, list):
                self.yleft = yrange[0]
                self.yright = yrange[1]
            elif isinstance(yrange, float):
                self.yleft = -1*np.abs(yrange)
                self.yright = np.abs(yrange)
        try:
            if self.yleft is None or self.yright is None:
                self.yleft = -1.0
                self.yright = 1.0
        except AttributeError:
            self.yleft = -1.0
            self.yright = 1.0
            print("Need to start with data or set yrange=[left,right]")

        self.hist = bh.Histogram(bh.axis.Regular(self.xbins, self.xleft, self.xright, metadata=self.xname),
                                 bh.axis.Regular(
                                     self.ybins, self.yleft, self.yright, metadata=self.yname),
                                 *args, **kwargs)
        self.color = None
        self.xs = np.linspace(self.xleft, self.xright, 5*self.xbins)
        self.ys = np.linspace(self.yleft, self.yright, 5*self.ybins)
        if xdata is not None and ydata is not None:
            self.hist.fill(xdata, ydata)

    @property
    def data(self):
        return self.hist

    def plot(self, ax=None,
             cmap=None, density: bool = True,  colorbar: bool = True, zeros: bool = True, log_cmap: bool = False):
        if not ax:
            ax = plt.gca()
        if density:
            # Compute the areas of each bin
            areas = functools.reduce(operator.mul, self.hist.axes.widths)
            # Compute the density
            zvalues = self.hist.view() / self.hist.sum() / areas
        else:
            zvalues = self.hist.view()

        zvalues = zvalues if zeros else np.where(zvalues == 0, np.nan,
                                                 zvalues)

        if log_cmap:
            if density:
                warnings.warn(
                    "WARNING: Using density = True with log_cmap can give weird results...")
            # handle zero value bins, which will show as empty otherwise (ugly!)
            zvalues[zvalues == 0] = 1E-30
            # dodge our dummy 'zero-substitute'
            vmin = np.min(zvalues[zvalues != 1E-30])
            vmax = np.max(zvalues)
            norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
        else:
            norm = None

        pc = ax.pcolormesh(*self.hist.axes.edges.T, zvalues.T, norm=norm,
                           cmap=cmap if cmap else 'viridis')

        ax.set_xlabel(self.xname)
        ax.set_ylabel(self.yname)
        ax.set_xlim([np.min(self.hist.axes[0]), np.max(self.hist.axes[0])])
        ax.set_ylim([np.min(self.hist.axes[1]), np.max(self.hist.axes[1])])
        if colorbar:
            plt.gcf().colorbar(pc, ax=ax, aspect=30)
        return pc

    def plot3D(self, ax=None,
               filled: bool = False, alpha: float = __ALPHA__,
               cmap=None, density: bool = True,  colorbar: bool = True, zeros: bool = True):
        fig = plt.gcf()
        ax = plt.axes(projection='3d')
        if not ax:
            ax = plt.gca()
        if density:
            # Compute the areas of each bin
            areas = functools.reduce(operator.mul, self.hist.axes.widths)
            # Compute the density
            zvalues = self.hist.view() / self.hist.sum() / areas
        else:
            zvalues = self.hist.view()

        zvalues = zvalues if zeros else np.where(zvalues == 0, np.nan,
                                                 zvalues)

        x, y = self.hist.axes.edges.T
        X, Y = np.meshgrid(x[:, 1:], y[1:, :])
        pc = ax.contour3D(X, Y, zvalues.T, 500,
                          cmap=cmap if cmap else 'viridis')

        ax.set_xlabel(self.xname)
        ax.set_ylabel(self.yname)
        if colorbar:
            plt.gcf().colorbar(pc, ax=ax, aspect=30)
        return pc

    def fill(self, data_x, data_y):
        # If we pass in a pandas series then rename the axes to the series name
        # Cool trick to not have to add labels to the histograms
        if isinstance(data_x, pd.Series):
            self.hist.axes[0].metadata = data_x.name
        if isinstance(data_y, pd.Series):
            self.hist.axes[1].metadata = data_y.name

        return self.hist.fill(data_x, data_y)

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

        x, y = self.hist.axes.edges.T
        areas = functools.reduce(operator.mul, self.hist.axes.widths)
        g = self.hist.view() / self.hist.sum() / areas
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

    def fitSliceX(self, ax=None, num_slices: int = 10, NSIMA: int = 3,
                  fit_range=None, center: bool = False, params=None,
                  plot: bool = True):
        if not ax:
            ax = plt.gca()
        if fit_range:
            slices = np.linspace(*fit_range, num_slices)
        else:
            slices = np.linspace(
                np.min(self.hist.axes[0]), np.max(self.hist.axes[0]), num_slices)
        width = np.abs(slices[0]-slices[1])
        outs = []
        xs = []
        yst = []
        ysb = []
        for sl in slices[:-1]:
            x_val = (sl + sl+width)/2
            try:
                slic = self.hist[bh.loc(sl):bh.loc(sl+width):bh.sum, :]
            except ValueError:
                continue

            temp_hist = Hist1D(boost_hist=slic)
            if center:
                params = GaussianModel().make_params()
                params['center'].set(value=0, min=-0.5, max=0.5)
                params['sigma'].set(value=0.1, min=0, max=1.0)

            try:
                out = temp_hist.fitGaussian(plots=False, params=params)
                outs.append(out)
                xs.append(x_val)
                yst.append(out.params['center'] + NSIMA * out.params['sigma'])
                ysb.append(out.params['center'] - NSIMA * out.params['sigma'])

            except TypeError:
                print(f"Cannot fit from [{sl}, {sl + width}]")
                continue
        top = Scatter(np.array(xs), np.array(yst))
        bot = Scatter(np.array(xs), np.array(ysb))
        return outs, top, bot

    def fitSliceY(self, ax=None, num_slices: int = 10, NSIMA: int = 3,
                  fit_range=None, center: bool = False, params=None,
                  plot: bool = True):
        if not ax:
            ax = plt.gca()
        if fit_range:
            slices = np.linspace(*fit_range, num_slices)
        else:
            slices = np.linspace(
                np.min(self.hist.axes[1]), np.max(self.hist.axes[1]), num_slices)
        width = np.abs(slices[0]-slices[1])
        outs = []
        ys = []
        xst = []
        xsb = []
        for sl in slices[:-1]:
            y_val = (sl + sl+width)/2
            try:
                slic = self.hist[:, bh.loc(sl):bh.loc(sl+width):bh.sum]
            except ValueError:
                continue

            temp_hist = Hist1D(boost_hist=slic)
            if center:
                params = GaussianModel().make_params()
                params['center'].set(value=0, min=-0.5, max=0.5)
                params['sigma'].set(value=0.1, min=0, max=1.0)

            try:
                out = temp_hist.fitGaussian(plots=False, params=params)
                outs.append(out)
                ys.append(y_val)
                xst.append(out.params['center'] + NSIMA * out.params['sigma'])
                xsb.append(out.params['center'] - NSIMA * out.params['sigma'])

            except TypeError:
                print(f"Cannot fit from [{sl}, {sl + width}]")
                continue
        left = Scatter(np.array(xst), np.array(ys))
        right = Scatter(np.array(xsb), np.array(ys))
        return outs, left, right
