from typing import List
import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from lmfit.models import *

__ALPHA__ = 0.8


class Hist1D(bh.Histogram):
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
                 *args, **kwargs) -> None:

        self.bins = bins
        self.name = name
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
        super(Hist1D, self).__init__(bh.axis.Regular(self.bins, self.left, self.right, metadata=self.name),
                                     *args, **kwargs)
        self.color = None
        self.model = None
        # If we started with data present then fill the histogram
        if data is not None:
            self.fill(data)
        # Make a set of xs for plotting lines with 5x the number of points from the bins
        self.xs = np.linspace(self.left, self.right, self.bins*5)

    def __getitem__(self, args):
        return "Slicing does not work...yet?"

    def histogram(self, ax=None, filled: bool = False, alpha: float = __ALPHA__,
                  color=None, density: bool = True, label: str = None):
        if not ax:
            ax = plt.gca()
        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']
        elif color:
            self.color = color

        x, y = self.hist_to_xy(density=density)

        if not label:
            label = self.axes[0].metadata

        st = ax.step(x, y, where='mid', color=self.color,
                     alpha=alpha,
                     label=None if filled else label)
        if filled:
            ys = self.view()/np.max(self.view()) if density else self.view()
            st = ax.fill_between(x, 0, ys,
                                 alpha=alpha,
                                 step='mid',
                                 color=self.color,
                                 label=label
                                 )
        if self.name:
            ax.set_xlabel(self.name)
        ax.legend()
        return st

    def errorbar(self, ax=None, alpha: float = __ALPHA__, color=None, density: bool = True, label=None):
        if not ax:
            ax = plt.gca()
        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']
        elif color:
            self.color = color

        x, y = self.hist_to_xy(density=density)

        label = label if label else self.axes[0].metadata

        st = ax.errorbar(x, y,
                         yerr=stats.sem(y),
                         fmt='.',
                         alpha=alpha,
                         color=self.color,
                         label=label)
        ax.set_xlabel(self.name)
        ax.legend()
        return st

    @property
    def x(self):
        return self.axes[0].centers

    @property
    def y(self):
        return self.view()/np.max(self.view())

    @property
    def y_counts(self):
        return self.view()

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

    def fill(self, data):
        # If we pass in a pandas series then rename the axes to the series name
        # Cool trick to not have to add labels to the histograms
        if isinstance(data, pd.Series):
            self.axes[0].metadata = data.name
        return super(Hist1D, self).fill(data)

    def fitGaussian(self, ax=None, alpha: float = __ALPHA__,
                    color=None, density: bool = True, params=None, plots: bool = True,
                    *args, **kwargs):
        self.model = GaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params, plots=plots)

    def customModel(self, model, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None):
        self.model = model
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def _fitModel(self, ax=None, alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None, plots: bool = True):
        if not ax:
            ax = plt.gca()
        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']
        elif color:
            self.color = color

        x, y = self.hist_to_xy(density=density)
        num_comp = len(self.model.components)

        # If we haven't set up params set them up now
        if params == None:
            pars = self.model.components[num_comp - 1].guess(y, x=x)
            for i in range(0, num_comp):
                pars.update(self.model.components[i].make_params())

        out = self.model.fit(y, params, x=x, nan_policy='omit')

        if num_comp > 1 and plots:
            comps = out.eval_components(x=self.xs)
            for name, comp in comps.items():
                ax.plot(self.xs, comp, label=name+"fit")
        if plots:
            ax.plot(self.xs, out.eval(x=self.xs),
                    label=self.model.name if num_comp == 1 else "Total Fit")

            ax.legend()
        return out

    def fitBreitWigner(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None,
                       *args, **kwargs):
        self.model = BreitWignerModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitComplexConstant(self, ax=None,
                           alpha: float = __ALPHA__, color=None,
                           density: bool = True, params=None,
                           *args, **kwargs):
        self.model = ComplexConstantModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitConstant(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None,
                    *args, **kwargs):
        self.model = ConstantModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitDampedHarmonicOscillator(self, ax=None,
                                    alpha: float = __ALPHA__, color=None,
                                    density: bool = True, params=None,
                                    *args, **kwargs):
        self.model = DampedHarmonicOscillatorModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitDampedOscillator(self, ax=None,
                            alpha: float = __ALPHA__, color=None,
                            density: bool = True, params=None,
                            *args, **kwargs):
        self.model = DampedOscillatorModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitDonaich(self, ax=None,
                   alpha: float = __ALPHA__, color=None,
                   density: bool = True, params=None,
                   *args, **kwargs):
        self.model = DonaichModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitDoniach(self, ax=None,
                   alpha: float = __ALPHA__, color=None,
                   density: bool = True, params=None,
                   *args, **kwargs):
        self.model = DoniachModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitExponentialGaussian(self, ax=None,
                               alpha: float = __ALPHA__, color=None,
                               density: bool = True, params=None,
                               *args, **kwargs):
        self.model = ExponentialGaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitExponential(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None,
                       *args, **kwargs):
        self.model = ExponentialModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitExpression(self, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None,
                      *args, **kwargs):
        self.model = ExpressionModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitLinear(self, ax=None,
                  alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None,
                  *args, **kwargs):
        self.model = LinearModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitLognormal(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     *args, **kwargs):
        self.model = LognormalModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitLorentzian(self, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None,
                      *args, **kwargs):
        self.model = LorentzianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitMoffat(self, ax=None,
                  alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None,
                  *args, **kwargs):
        self.model = MoffatModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitParabolic(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     *args, **kwargs):
        self.model = ParabolicModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitPearson7(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None,
                    *args, **kwargs):
        self.model = Pearson7Model(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitPolynomial(self, degree=5, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None,
                      *args, **kwargs):
        self.model = PolynomialModel(degree, *args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitPowerLaw(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None,
                    *args, **kwargs):
        self.model = PowerLawModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitPseudoVoigt(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None,
                       *args, **kwargs):
        self.model = PseudoVoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitQuadratic(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     *args, **kwargs):
        self.model = QuadraticModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitRectangle(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     *args, **kwargs):
        self.model = RectangleModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitSkewedGaussian(self, ax=None,
                          alpha: float = __ALPHA__, color=None,
                          density: bool = True, params=None,
                          *args, **kwargs):
        self.model = SkewedGaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitSkewedVoigt(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None,
                       *args, **kwargs):
        self.model = SkewedVoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitSplitLorentzian(self, ax=None,
                           alpha: float = __ALPHA__, color=None,
                           density: bool = True, params=None,
                           *args, **kwargs):
        self.model = SplitLorentzianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitStep(self, ax=None,
                alpha: float = __ALPHA__, color=None,
                density: bool = True, params=None,
                *args, **kwargs):
        self.model = StepModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitStudentsT(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     *args, **kwargs):
        self.model = StudentsTModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitThermalDistribution(self, ax=None,
                               alpha: float = __ALPHA__, color=None,
                               density: bool = True, params=None,
                               *args, **kwargs):
        self.model = ThermalDistributionModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def fitVoigt(self, ax=None,
                 alpha: float = __ALPHA__, color=None,
                 density: bool = True, params=None,
                 *args, **kwargs):
        self.model = VoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)
