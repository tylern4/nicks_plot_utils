import boost_histogram as bh
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
import pandas as pd
from lmfit.models import *

__ALPHA__ = 0.8


class Scatter:
    def __init__(self, x, y,
                 name: str = None,
                 *args, **kwargs) -> None:
        self._x = x
        self._y = y
        self.color = None
        if isinstance(x, pd.Series):
            self.x_name = x.name
        if isinstance(x, pd.Series):
            self.y_name = y.name
        self.name = name
        # Give fit range +- 1%
        self.xs = np.linspace(np.min(self._x)*0.09, np.max(self._x)*1.01, 500)

    def errorbar(self, ax=None, alpha: float = __ALPHA__, color=None,
                 density: bool = True, label=None):
        if not ax:
            ax = plt.gca()

        if color:
            self.color = color
        else:
            self.color = next(ax._get_lines.prop_cycler)['color']

        label = label if label else self.name

        st = ax.errorbar(self._x, self._y,
                         yerr=stats.sem(self.y),
                         fmt='.',
                         alpha=alpha,
                         color=self.color,
                         label=label, linewidth=0)
        try:
            ax.set_xlabel(self.x_name)
            ax.set_ylabel(self.y_name)
            ax.set_title(self.name)
        except:
            pass

        if label:
            ax.legend()
        return st

    def scatter(self, ax=None, alpha: float = __ALPHA__, color=None,
                density: bool = True, label=None):
        if not ax:
            ax = plt.gca()

        if color:
            self.color = color
        else:
            self.color = next(ax._get_lines.prop_cycler)['color']

        label = label if label else self.name

        st = ax.scatter(self._x, self._y,
                        alpha=alpha,
                        color=self.color,
                        label=label)
        try:
            ax.set_xlabel(self.x_name)
            ax.set_ylabel(self.y_name)
            ax.set_title(self.name)
        except:
            pass
        if label:
            ax.legend()
        return st

    def histogram(self, ax=None, filled: bool = False,
                  alpha: float = __ALPHA__, fill_alpha: float = None,
                  color=None, label: str = None, factor: int = 1.0, loc='best'):
        if not ax:
            ax = plt.gca()
        if color:
            self.color = color
        else:
            self.color = next(ax._get_lines.prop_cycler)['color']

        x = self._x
        y = self._y
        # Height factor to change max of density plots
        y *= factor

        if not label:
            label = self.name
            if isinstance(label, dict):
                label = None

        st = ax.step(x, y, where='mid', color=self.color,
                     alpha=alpha,
                     label=None if filled else label)

        # if non-zero counts at the edge of a histogram, draw vertical lines at limits
        # 0 is left edge -1 is right edge
        for edge in [0, -1]:
            if y[edge] != 0:
                ax.vlines(x[edge], 0, y[edge], color=self.color, alpha=alpha)

        filled = filled if fill_alpha is None else True

        if filled:
            # If filled not defined set it to lines alpha - 0.1
            fill_alpha = fill_alpha if fill_alpha is not None else alpha - 0.1

            self._y *= factor
            st = ax.fill_between(x, 0, self._y,
                                 alpha=fill_alpha,
                                 step='mid',
                                 color=self.color,
                                 label=label,
                                 )
        if self.name:
            ax.set_title(self.name)
        if label:
            ax.legend(loc=loc)
        return st

    @property
    def x(self):
        """I'm the 'x' property."""
        return self._x

    @x.setter
    def x(self, value):
        self._x = np.array(value)

    @property
    def y(self):
        """I'm the 'y' property."""
        return self._y

    @y.setter
    def y(self, value):
        self._y = np.array(value)

    def fitGaussian(self, ax=None, alpha: float = __ALPHA__,
                    color=None, density: bool = True, params=None,
                    plots: bool = True, method=None,
                    * args, **kwargs):
        self.model = GaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def customModel(self, model, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None,
                    plots: bool = False, method=None):
        self.model = model
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def _fitModel(self, ax=None, alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None, plots: bool = True, method=None):
        if not ax:
            ax = plt.gca()

        if color:
            self.color = color
        else:
            self.color = next(ax._get_lines.prop_cycler)['color']

        num_comp = len(self.model.components)

        # If we haven't set up params set them up now
        if params is None:
            params = self.model.components[num_comp -
                                           1].guess(self._y, x=self._x)
            for i in range(1, num_comp):
                params.update(self.model.components[i].make_params())
        method = method if method else 'leastsq'

        out = self.model.fit(self._y, params, x=self._x,
                             nan_policy='omit', method=method)

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
                       plots: bool = True, method=None,
                       *args, **kwargs):
        self.model = BreitWignerModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitComplexConstant(self, ax=None,
                           alpha: float = __ALPHA__, color=None,
                           density: bool = True, params=None,
                           plots: bool = True, method=None,
                           *args, **kwargs):
        self.model = ComplexConstantModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitConstant(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None,
                    plots: bool = True, method=None,
                    *args, **kwargs):
        self.model = ConstantModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitDampedHarmonicOscillator(self, ax=None,
                                    alpha: float = __ALPHA__, color=None,
                                    density: bool = True, params=None,
                                    plots: bool = True, method=None,
                                    *args, **kwargs):
        self.model = DampedHarmonicOscillatorModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitDampedOscillator(self, ax=None,
                            alpha: float = __ALPHA__, color=None,
                            density: bool = True, params=None,
                            plots: bool = True, method=None,
                            *args, **kwargs):
        self.model = DampedOscillatorModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitDonaich(self, ax=None,
                   alpha: float = __ALPHA__, color=None,
                   density: bool = True, params=None,
                   plots: bool = True, method=None,
                   *args, **kwargs):
        self.model = DonaichModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitDoniach(self, ax=None,
                   alpha: float = __ALPHA__, color=None,
                   density: bool = True, params=None,
                   plots: bool = True, method=None,
                   *args, **kwargs):
        self.model = DoniachModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitExponentialGaussian(self, ax=None,
                               alpha: float = __ALPHA__, color=None,
                               density: bool = True, params=None,
                               plots: bool = True, method=None,
                               *args, **kwargs):
        self.model = ExponentialGaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitExponential(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None,
                       plots: bool = True, method=None,
                       *args, **kwargs):
        self.model = ExponentialModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitExpression(self, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None,
                      plots: bool = True, method=None,
                      *args, **kwargs):
        self.model = ExpressionModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitLinear(self, ax=None,
                  alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None,
                  plots: bool = True, method=None,
                  *args, **kwargs):
        self.model = LinearModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitLognormal(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     plots: bool = True, method=None,
                     *args, **kwargs):
        self.model = LognormalModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitLorentzian(self, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None,
                      plots: bool = True, method=None,
                      *args, **kwargs):
        self.model = LorentzianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitMoffat(self, ax=None,
                  alpha: float = __ALPHA__, color=None,
                  density: bool = True, params=None,
                  plots: bool = True, method=None,
                  *args, **kwargs):
        self.model = MoffatModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitParabolic(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     plots: bool = True, method=None,
                     *args, **kwargs):
        self.model = ParabolicModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitPearson7(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None,
                    plots: bool = True, method=None,
                    *args, **kwargs):
        self.model = Pearson7Model(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitPolynomial(self, degree=5, ax=None,
                      alpha: float = __ALPHA__, color=None,
                      density: bool = True, params=None,
                      plots: bool = True, method=None,
                      *args, **kwargs):
        self.model = PolynomialModel(degree, *args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitPowerLaw(self, ax=None,
                    alpha: float = __ALPHA__, color=None,
                    density: bool = True, params=None,
                    plots: bool = True, method=None,
                    *args, **kwargs):
        self.model = PowerLawModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitPseudoVoigt(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None,
                       plots: bool = True, method=None,
                       *args, **kwargs):
        self.model = PseudoVoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitQuadratic(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     plots: bool = True, method=None,
                     *args, **kwargs):
        self.model = QuadraticModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitRectangle(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     plots: bool = True, method=None,
                     *args, **kwargs):
        self.model = RectangleModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitSkewedGaussian(self, ax=None,
                          alpha: float = __ALPHA__, color=None,
                          density: bool = True, params=None,
                          plots: bool = True, method=None,
                          *args, **kwargs):
        self.model = SkewedGaussianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitSkewedVoigt(self, ax=None,
                       alpha: float = __ALPHA__, color=None,
                       density: bool = True, params=None,
                       plots: bool = True, method=None,
                       *args, **kwargs):
        self.model = SkewedVoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitSplitLorentzian(self, ax=None,
                           alpha: float = __ALPHA__, color=None,
                           density: bool = True, params=None,
                           plots: bool = True, method=None,
                           *args, **kwargs):
        self.model = SplitLorentzianModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitStep(self, ax=None,
                alpha: float = __ALPHA__, color=None,
                density: bool = True, params=None,
                plots: bool = True, method=None,
                *args, **kwargs):
        self.model = StepModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitStudentsT(self, ax=None,
                     alpha: float = __ALPHA__, color=None,
                     density: bool = True, params=None,
                     plots: bool = True, method=None,
                     *args, **kwargs):
        self.model = StudentsTModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitThermalDistribution(self, ax=None,
                               alpha: float = __ALPHA__, color=None,
                               density: bool = True, params=None,
                               plots: bool = True, method=None,
                               *args, **kwargs):
        self.model = ThermalDistributionModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)

    def fitVoigt(self, ax=None,
                 alpha: float = __ALPHA__, color=None,
                 density: bool = True, params=None,
                 plots: bool = True, method=None,
                 *args, **kwargs):
        self.model = VoigtModel(*args, **kwargs)
        return self._fitModel(ax=ax, alpha=alpha, color=color,
                              density=density, params=params,
                              plots=plots, method=method)
