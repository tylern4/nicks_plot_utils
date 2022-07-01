#!/usr/bin/env python
# coding: utf-8

import pylandau
import numpy as np
import pytest
from nicks_plot_utils import Hist1D, Scatter
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import *
import boost_histogram as bh

mu = 0.0
sigma = 0.2
num = 10_000_000

data = pd.DataFrame({'W': np.random.gumbel(mu, sigma, num),
                     'y': np.random.normal(mu, 0.1, num),
                     'z': np.random.normal(0.5, 0.5, num)})


def test_fitGaussian():
    testHist = Hist1D(data=data.W, name="$\\varphi$")
    testHist.histogram(fill_alpha=0.3, linewidth=3, filled=True)
    fit = testHist.fitGaussian(fit_range=[-0.2, 0.2])

    assert fit.success == True
    assert fit.best_values['center'] == pytest.approx(mu, abs=1e-1)
    assert fit.best_values['sigma'] == pytest.approx(sigma, abs=1e-1)


def test_custom_function():
    # # Make a landau from pylandau
    mpv, eta, sigma, A = 30, 5, 4, 1000
    x = np.arange(0, 100, 0.5)
    y = pylandau.landau(x, mpv, eta, A)

    # # Put it into plot utils as a scatter object
    land = Scatter(x, y)

    def landau(x, mpv=0, eta=1, A=1):
        return pylandau.landau(x, mpv=mpv, eta=eta, A=A)

    # # Make a custom model and params
    mod = Model(landau, independent_vars=['x'])
    pars = mod.make_params()
    pars['mpv'].set(value=x[np.argmax(y)])
    pars['eta'].set(value=4.5, min=1, max=10)
    pars['A'].set(value=np.max(y), min=1)

    # # Fit and draw the model
    fit = land.customModel(mod, params=pars)
    assert fit.success == True
    assert fit.best_values['mpv'] == pytest.approx(mpv, abs=1e-1)
    assert fit.best_values['eta'] == pytest.approx(eta, abs=1e-1)
    assert fit.best_values['A'] == pytest.approx(A, abs=1e-1)


def test_from_boost_hist():
    bins = 200
    total = 10_000
    left = -1
    right = 1
    name = "test"
    bhhist = bh.Histogram(bh.axis.Regular(bins, left, right, metadata=name))
    bhhist.fill(np.random.normal(0.0, 2.0, total))
    hist = Hist1D(boost_hist=bhhist)

    assert hist.bins == bins
    assert len(hist.data.to_numpy()[0]) == bins


def test_fill_hist():
    bins = 200
    total = 10_000
    left = -1
    right = 1
    hist = Hist1D(xrange=[left, right], bins=bins)
    hist.fill(np.random.normal(0.0, 2.0, total))

    assert hist.bins == bins
    assert len(hist.data.to_numpy()[0]) == bins

    hist = Hist1D(xrange=right, bins=bins)
    hist.fill(np.random.normal(0.0, 2.0, total))

    hist.slice_to_xy(slice_range=[-0.5, 0.5])
    hist.hist_to_xy()

    hist.errorbar()

    assert hist.bins == bins
    assert hist.left == left
    assert hist.right == right


def test_fits():
    bins = 200
    total = 10_000
    left = -1
    right = 1
    hist = Hist1D(xrange=[left, right], bins=bins)
    hist.fill(np.random.normal(0.0, 2.0, total))

    hist.x
    hist.y
    hist.y_counts

    hist.fitBreitWigner()
    hist.fitDampedHarmonicOscillator()
    hist.fitDampedOscillator()
    hist.fitDoniach()
    hist.fitExponential()
    hist.fitLinear()
    hist.fitLognormal()
    hist.fitLorentzian()
    hist.fitMoffat()
    hist.fitParabolic()
    hist.fitPearson7()
    hist.fitPseudoVoigt()
    hist.fitQuadratic()
    hist.fitRectangle()
    hist.fitSkewedGaussian()
    hist.fitSkewedVoigt()
    hist.fitSplitLorentzian()
    hist.fitStep()
    hist.fitStudentsT()
    hist.fitThermalDistribution()
    hist.fitVoigt()
