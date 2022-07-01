#!/usr/bin/env python
# coding: utf-8

import pylandau
import numpy as np
import pytest
from nicks_plot_utils import Hist2D
import matplotlib.pyplot as plt
import pandas as pd
from lmfit.models import *
import boost_histogram as bh
import lmfit

mu = 0.0
sigma = 0.2
num = 10_000_000

data = pd.DataFrame({'W': np.random.gumbel(mu, sigma, num),
                     'y': np.random.normal(mu, 0.1, num),
                     'z': np.random.normal(0.5, 0.5, num)})


def test_fitGaussian():
    testHist = Hist2D(xdata=data['y'], ydata=data['z'])
    testHist.fitGausian()
    testHist.plot()


def test_slices():
    testHist = Hist2D(xdata=data['y'], ydata=data['z'])
    x = testHist.fitSliceX(num_slices=10)
    assert isinstance(x[0][0], lmfit.model.ModelResult)


def test_fill():
    testHist = Hist2D(xbins=100, ybins=100, xrange=[-1, 1], yrange=[-1, 1])
    num = 10000
    testHist.fill(np.random.normal(0.5, 0.5, num),
                  np.random.normal(0.5, 0.5, num))
    testHist.hist
