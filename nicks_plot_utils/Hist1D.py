from typing import List
import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from lmfit.models import *

__ALPHA__ = 0.8


class Hist1D:
    """Hist1D is a wrapper around a boost histogram which sets up a 1D histogram and adds simples plots and fitting.

    Args:
        [Required]
            bins -> int : Sets the number of bins to use
        [Semi-Required]
            xrange -> List [Required unless data is suplied)] Sets the xrange of the histogram
        [Optional]
            data -> Array like : Data to fill the histogram with
            name -> string : Name to be used with automatic axes labes

        *args, **kwargs are passed to bh.Histogram

    """

    def __init__(self,
                 data=None,
                 xrange: List = None,
                 bins: float = 100,
                 name: str = None,
                 boost_hist=None,
                 *args, **kwargs) -> None:

        self.name = name

        if boost_hist is not None:
            self.hist = boost_hist
            # set things for plotting:
            self.bins = len(self.hist.axes[0].centers)
            self.left = min(self.hist.axes[0].edges)
            self.right = max(self.hist.axes[0].edges)
            self.xs = np.linspace(self.left, self.right, self.bins*5)
            self.color = None
            return

        self.bins = bins
        # Get the left and right bin edges either from range or dataset with fallbacks
        if data is not None:
            self.left = np.min(data)
            self.right = np.max(data)
        if xrange is not None:
            if isinstance(xrange, list):
                self.left = xrange[0]
                self.right = xrange[1]
            elif isinstance(xrange, float):
                self.left = -1*np.abs(xrange)
                self.right = np.abs(xrange)
        try:
            if self.left is None or self.right is None:
                self.left = -1.0
                self.right = 1.0
        except AttributeError:
            self.left = -1.0
            self.right = 1.0
            print("Need to start with data or set xrange=[left,right]")

        # Once we have the bins and edges we can make out boost histogram object
        self.hist = bh.Histogram(bh.axis.Regular(self.bins, self.left, self.right, metadata=self.name),
                                 *args, **kwargs)
        self.color = None
        self.model = None
        # If we started with data present then fill the histogram
        if data is not None:
            self.hist.fill(data)
        # Make a set of xs for plotting lines with 5x the number of points from the bins
        self.xs = np.linspace(self.left, self.right, self.bins*5)

    @property
    def data(self):
        return self.hist

    def histogram(self, ax=None, filled: bool = False, alpha: float = __ALPHA__, fill_alpha: float = None,
                  color=None, density: bool = True, label: str = None, factor: int = 1.0, loc='best',
                  *args, **kwargs):
        if not ax:
            ax = plt.gca()
        if color:
            self.color = color
        else:
            self.color = None

        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']

        x, y = self.hist_to_xy(density=density)
        # Height factor to change max of density plots
        y *= factor

        if not label:
            label = self.hist.axes[0].metadata
            if isinstance(label, dict):
                label = None

        st = ax.step(x, y, where='mid', color=self.color,
                     alpha=alpha,
                     label=None if filled else label,
                     *args, **kwargs)

        # if non-zero counts at the edge of a histogram, draw vertical lines at limits
        # 0 is left edge -1 is right edge
        for edge in [0, -1]:
            if y[edge] != 0:
                step_lw = plt.getp(st[0], 'linewidth')
                step_ls = plt.getp(st[0], 'linestyle')
                ax.vlines(x[edge], 0, y[edge], color=self.color,
                          alpha=alpha, lw=step_lw, ls=step_ls)

        filled = filled if fill_alpha is None else True

        if filled:
            # If filled not defined set it to lines alpha - 0.1
            fill_alpha = fill_alpha if fill_alpha is not None else alpha - 0.1

            ys = self.hist.values()/np.max(self.hist.values()) if density else self.hist.values()
            ys *= factor
            st = ax.fill_between(x, 0, ys,
                                 alpha=fill_alpha,
                                 step='mid',
                                 color=self.color,
                                 label=label,
                                 )
        if self.name:
            ax.set_xlabel(self.name)
        if label:
            ax.legend(loc=loc)
        return st

    def errorbar(self, ax=None, alpha: float = __ALPHA__,
                 color=None, density: bool = True, label=None,
                 errorcalc=None, factor: int = 1.0):
        if not ax:
            ax = plt.gca()

        if color:
            self.color = color
        else:
            self.color = None

        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']

        x, y = self.hist_to_xy(density=density)
        y *= factor

        label = label if label else (
            self.hist.axes[0].metadata if not isinstance(self.hist.axes[0].metadata, dict) else None)

        if type(errorcalc) is float or type(errorcalc) is int or type(errorcalc) is np.float64:
            # If we have a number make array with the same number the same size as y
            yerr = np.ones_like(y) * errorcalc
        elif type(errorcalc) is np.ndarray:
            # pass in errors from nd with errors
            # Had to put these at the top to get rid of element wise comparison problems in numpy
            if errorcalc.size == y.size:
                # Check that they are the same size as the y values or errorbar will break
                yerr = errorcalc
            else:
                print("Zero error bars assigned, ndarray and y are not the same size.")
                yerr = np.zeros_like(y)
        elif errorcalc is None or errorcalc == "" or errorcalc == "sem":
            # Default to standard error of mean for errorbars
            yerr = stats.sem(y)
        elif errorcalc == "std":
            # Use standard deviation
            yerr = np.std(y)
        elif errorcalc == "sqrt":
            # Use sqrt(N) for errors
            # Need more testing to make sure it follows statistics for density plots
            yerr = np.sqrt(y)
        elif callable(errorcalc):
            # If we send in a function use that to compute the errors
            yerr = errorcalc(y)
        else:
            yerr = np.zeros_like(y)

        st = ax.errorbar(x, y,
                         yerr=yerr,
                         fmt='.',
                         alpha=alpha,
                         color=self.color,
                         label=label)
        ax.set_xlabel(self.name)
        if label:
            ax.legend()
        return st

    @property
    def x(self):
        return self.hist.axes[0].centers

    @property
    def y(self):
        return self.hist.values()/np.max(self.hist.values())

    @property
    def y_counts(self):
        return self.hist.values()

    def hist_to_xy(self, density: bool = True):
        """Takes a histogram and makes it into a scatter of x,y
        Useful for plotting in different ways and for fitting

        Args:
            density (bool, optional): Choose to plot y values or density of y values. Defaults to True.

        Returns:
            Tuple(x, y): Returns a tuple of np arrays for x and y values of histogram
        """
        # Check if we want density and set y accordingly
        try:
            y = self.y if density else self.y_counts
        except:
            y = self.y_counts

        return (self.x, y)

    def slice_to_xy(self, slice_range, density: bool = True):
        """Return the x, y values from a histogram in the range
        Useful for plotting in different ways and for fitting

        Args:
            slice_range (Required) : Range to return x, y slice from
            density (bool, optional): Choose to plot y values or density of y values. Defaults to True.

        Returns:
            Tuple(x, y): Returns a tuple of np arrays for x and y values of histogram
        """
        slic = self.hist[bh.loc(slice_range[0]):bh.loc(slice_range[1])]

        x = slic.axes[0].centers
        if density:
            y = slic.values()/np.max(slic.values())
        else:
            y = slic.values()

        return (x, y)

    def fill(self, data):
        # If we pass in a pandas series then rename the axes to the series name
        # Cool trick to not have to add labels to the histograms
        if isinstance(data, pd.Series):
            self.hist.axes[0].metadata = data.name
        return self.hist.fill(data)

    def fitGaussian(self, ax=None, alpha: float = __ALPHA__, fit_range=None,
                    color=None, density: bool = True, params=None, plots: bool = True,

                    *args, **kwargs):
        self.model = GaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, plots=plots, fit_range=fit_range)

    def customModel(self, model, ax=None,
                    alpha: float = __ALPHA__, color=None, density: bool = True,
                    params=None, fit_range=None, weights=None, plots: bool = True):
        self.model = model
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range, plots=plots, weights=weights)

    def _fitModel(self, ax=None, alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None, plots: bool = True, fit_range=None,
                  weights=None, label=None, loc='best'):

        if not ax:
            ax = plt.gca()

        try:
            self.xs
        except AttributeError:
            _min = 0
            _max = self.hist.shape[0]
            _axs = self.hist.axes[0]
            self.xs = np.linspace(_axs.edges[_min], _axs.edges[_max], 500)

        if color:
            self.color = color
        else:
            self.color = None

        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']

        plot_xs = self.xs

        if fit_range is None:
            x, y = self.hist_to_xy(density=density)
        else:
            x, y = self.slice_to_xy(slice_range=fit_range, density=density)
            plot_xs = np.linspace(x[0], x[-1], self.bins*5)

        num_comp = len(self.model.components)

        # If we haven't set up params set them up now
        if params is None:
            pars = self.model.components[num_comp - 1].guess(y, x=x)
            for i in range(0, num_comp):
                pars.update(self.model.components[i].make_params())

        out = self.model.fit(
            y, params, x=x, nan_policy='omit', weights=weights)

        if num_comp > 1 and plots:
            comps = out.eval_components(x=plot_xs)
            for name, comp in comps.items():
                ax.plot(plot_xs, comp, label=name+"fit", zorder=3)
        if plots:
            ax.plot(plot_xs, out.eval(x=plot_xs),
                    label=self.model.name if num_comp == 1 else "Total Fit", zorder=3, lw=3)
            if label:
                ax.legend(loc=loc)

        return out

    def fitBreitWigner(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None, fit_range=None,
                       *args, **kwargs):
        self.model = BreitWignerModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitComplexConstant(self, ax=None,
                           alpha: float = __ALPHA__, color=None,
                           density: bool = True, params=None, fit_range=None,
                           *args, **kwargs):
        self.model = ComplexConstantModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitConstant(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None, fit_range=None,
                    *args, **kwargs):
        self.model = ConstantModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitDampedHarmonicOscillator(self, ax=None,
                                    alpha: float = __ALPHA__, color=None,
                                    density: bool = True, params=None, fit_range=None,
                                    *args, **kwargs):
        self.model = DampedHarmonicOscillatorModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitDampedOscillator(self, ax=None,
                            alpha: float = __ALPHA__, color=None,
                            density: bool = True, params=None, fit_range=None,
                            *args, **kwargs):
        self.model = DampedOscillatorModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitDoniach(self, ax=None,
                   alpha: float = __ALPHA__, color=None,
                   density: bool = True, params=None, fit_range=None,
                   *args, **kwargs):
        self.model = DoniachModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitExponentialGaussian(self, ax=None,
                               alpha: float = __ALPHA__, color=None,
                               density: bool = True, params=None, fit_range=None,
                               *args, **kwargs):
        self.model = ExponentialGaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitExponential(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None, fit_range=None,
                       *args, **kwargs):
        self.model = ExponentialModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitExpression(self, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None, fit_range=None,
                      *args, **kwargs):
        self.model = ExpressionModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitLinear(self, ax=None,
                  alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None, fit_range=None,
                  *args, **kwargs):
        self.model = LinearModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitLognormal(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None, fit_range=None,
                     *args, **kwargs):
        self.model = LognormalModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitLorentzian(self, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None, fit_range=None,
                      *args, **kwargs):
        self.model = LorentzianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitMoffat(self, ax=None,
                  alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None, fit_range=None,
                  *args, **kwargs):
        self.model = MoffatModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitParabolic(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None, fit_range=None,
                     *args, **kwargs):
        self.model = ParabolicModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitPearson7(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None, fit_range=None,
                    *args, **kwargs):
        self.model = Pearson7Model(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitPolynomial(self, degree=5, ax=None,
                      alpha: float = __ALPHA__, color=None, fit_range=None,
                      density: bool = True, params=None,
                      *args, **kwargs):
        self.model = PolynomialModel(degree, *args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitPowerLaw(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None, fit_range=None,
                    *args, **kwargs):
        self.model = PowerLawModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitPseudoVoigt(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None, fit_range=None,
                       *args, **kwargs):
        self.model = PseudoVoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitQuadratic(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None, fit_range=None,
                     *args, **kwargs):
        self.model = QuadraticModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitRectangle(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None, fit_range=None,
                     *args, **kwargs):
        self.model = RectangleModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitSkewedGaussian(self, ax=None,
                          alpha: float = __ALPHA__, color=None,
                          density: bool = True, params=None, fit_range=None,
                          *args, **kwargs):
        self.model = SkewedGaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitSkewedVoigt(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None, fit_range=None,
                       *args, **kwargs):
        self.model = SkewedVoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitSplitLorentzian(self, ax=None,
                           alpha: float = __ALPHA__, color=None,
                           density: bool = True, params=None, fit_range=None,
                           *args, **kwargs):
        self.model = SplitLorentzianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitStep(self, ax=None,
                alpha: float = __ALPHA__, color=None,
                density: bool = True, params=None, fit_range=None,
                *args, **kwargs):
        self.model = StepModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitStudentsT(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None, fit_range=None,
                     *args, **kwargs):
        self.model = StudentsTModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitThermalDistribution(self, ax=None,
                               alpha: float = __ALPHA__, color=None,
                               density: bool = True, params=None, fit_range=None,
                               *args, **kwargs):
        self.model = ThermalDistributionModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)

    def fitVoigt(self, ax=None,
                 alpha: float = __ALPHA__, color=None,
                 density: bool = True, params=None, fit_range=None,
                 *args, **kwargs):
        self.model = VoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, fit_range=fit_range)
