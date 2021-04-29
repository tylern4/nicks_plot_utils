"""
nicks_plot_utils

A whole bunch of plots I constantly make made into a package
"""
from .Hist1D import Hist1D
from .Hist2D import Hist2D
from .Scatter import Scatter
from .Rhist import TH1D, TH2D
import matplotlib.pyplot as plt

plt.style.use('bmh')
# plt.style.use('seaborn-colorblind')

plt.rcParams['axes.formatter.use_mathtext'] = True
plt.rcParams['font.size'] = 16
plt.rcParams['axes.labelsize'] = 20
plt.rcParams['axes.titlesize'] = 20
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.figsize'] = [16, 9]


__version__ = "1.0.0"
__author__ = 'Nick Tyler'
__credits__ = 'Nick Tyler'

__all__ = ["Hist1D", "Hist2D", "Scatter", "TH1D", "TH2D"]
