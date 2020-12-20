import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from lmfit.models import *

__ALPHA__ = 0.8


class Hist1D(bh.Histogram):
    def __init__(self,
                 left: float = -1,
                 right: float = 1,
                 bins: float = 100,
                 name: str = None,
                 *args, **kwargs) -> None:
        super(Hist1D, self).__init__(bh.axis.Regular(bins, left, right, metadata=name),
                                     *args, **kwargs)
        self.color = None
        self.name = name
        self.model = GaussianModel()
        self.left = left
        self.right = right
        self.xs = np.linspace(left, right, 500)

    def histogram(self, ax=None, filled: bool = False, alpha: float = __ALPHA__, color=None, density: bool = True):
        if not ax:
            ax = plt.gca()
        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']
        elif color:
            self.color = color

        x, y = self.hist_to_xy(density=density)
        st = ax.step(x, y, where='post', color=self.color,
                     alpha=alpha,
                     label=None if filled else self.axes[0].metadata)
        if filled:
            ys = self.view()/np.max(self.view()) if density else self.view()
            st = ax.fill_between(x, 0, ys,
                                 alpha=alpha,
                                 step='post',
                                 color=self.color,
                                 label=self.axes[0].metadata
                                 )
        if self.name:
            ax.set_xlabel(self.name)
        ax.legend()
        return st

    def errorbar(self, ax=None, alpha: float = __ALPHA__, color=None, density: bool = True):
        if not ax:
            ax = plt.gca()
        if not self.color:
            self.color = next(ax._get_lines.prop_cycler)['color']
        elif color:
            self.color = color

        x, y = self.hist_to_xy(density=density)

        st = ax.errorbar(x, y,
                         yerr=stats.sem(y),
                         fmt='.',
                         alpha=alpha,
                         color=self.color,
                         label=self.axes[0].metadata)
        ax.set_xlabel(self.name)
        ax.legend()
        return st

    def hist_to_xy(self, density: bool = True):
        """Takes a histogram and makes it into a scatter of x,y
        Useful for plotting in different ways and for fitting

        Args:
            density (bool, optional): Choose to plot y values or density of y values. Defaults to True.

        Returns:
            Tuple(x, y): Returns a tuple of np arrays for x and y values of histogram
        """
        # Get bin centers for x value
        x = self.axes[0].centers
        # Check if we want density and set y accordingly
        try:
            y = self.view()/np.max(self.view()) if density else self.view()
        except:
            y = self.view()

        return (x, y)

    def fill(self, data):
        # If we pass in a pandas series then rename the axes to the series name
        # Cool trick to not have to add labels to the histograms
        if isinstance(data, pd.Series):
            self.axes[0].metadata = data.name
        return super(Hist1D, self).fill(data)

    def fitGaussian(self, ax=None, alpha: float = __ALPHA__,
                    color=None, density: bool = True, params=None,
                    *args, **kwargs):
        self.model = GaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def customModel(self, model, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None):
        self.model = model
        return self._fitModel(ax=ax, alpha=alpha, color=color, density=density, params=params)

    def _fitModel(self, ax=None, alpha: float = __ALPHA__, color=None, density: bool = True, params=None):
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

        if num_comp > 1:
            comps = out.eval_components(x=self.xs)
            for name, comp in comps.items():
                ax.plot(self.xs, comp, label=name+"fit")

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
