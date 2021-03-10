This repo is forked from [tylern4/nicks_plot_utils](https://github.com/tylern4/nicks_plot_utils) for the sake of adding some personal-choice changes. 

## Ploting utils for myself
----------------------------

In trying to decrease the amount of copy paste in my python code I decided to put some of plotting I do commonly into a few classes. They are wrappers around the [boost-histogram](https://boost-histogram.readthedocs.io/en/latest/) package so I can fill, plot, and then fit my most common histograms I make. Fitting is done useing [lmfit](https://lmfit.github.io/lmfit-py/) and I've made wrappers for the built in models for one line easy fitting as well as a way to use custom models for more complex fits.

### Examples

[One Dimentional Histogram](Examples/Example_Hist1D.ipynb)

[Two Dimentional Histogram](Examples/Example_Hist2D.ipynb)

[uproot](Examples/uproot.ipynb)
