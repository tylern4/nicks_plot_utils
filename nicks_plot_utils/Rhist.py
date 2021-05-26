import numpy as np
from . import Scatter

# Add a fill method that handles array-like for root
try:
    import ROOT

    class TH2D(ROOT.TH2D):
        def __init__(self, *args):
            ROOT.TH2D.__init__(self, *args)

        def fill(self, xs, ys, weight=1):
            # If the values object we put in has the length property fill in each with loop
            if hasattr(xs, "__len__"):
                # Check if weights is an array as well and if not make an array of ones
                weight = weight if hasattr(
                    weight, "__len__") else np.ones_like(xs) * weight
                # Use ROOTs FillN function to loop in C++ instead of python
                weight = np.array(weight)
                xs = np.array(xs)
                ys = np.array(ys)
                N = xs.shape[0]
                self.FillN(N, xs, ys, weight, 1)
            else:
                self.Fill(xs, ys, weight)

    class TH1D(ROOT.TH1D):
        def __init__(self, *args):
            ROOT.TH1D.__init__(self, *args)

        def fill(self, val, weight=1.0):
            """Fill an array into ROOT histogram

            Args:
                val (Array like): Array of values to fill into histogram
                weight (Array or Float, optional): Array of weights to fill for each event. Defaults to 1.
            """
            # If the values object we put in has the length property fill in each with loop
            if hasattr(val, "__len__"):
                # Check if weights is an array as well and if not make an array of ones
                weight = weight if hasattr(
                    weight, "__len__") else np.ones_like(val) * weight
                # Use ROOTs FillN function to loop in C++ instead of python
                weight = np.array(weight)
                val = np.array(val)
                N = val.shape[0]
                self.FillN(N, val, weight, 1)
            else:
                self.Fill(val, weight)

        def _get_xy(self):
            self.y = np.array(self)[1:-1]
            self.x = np.linspace(self.GetXaxis().GetXmin(),
                                 self.GetXaxis().GetXmax()-self.GetXaxis().GetBinWidth(1),
                                 num=self.GetNbinsX())

        def toScatter(self, name: str = None, *args, **kwargs):
            self._get_xy()
            name = name if name else self.GetTitle()
            return Scatter(self.x, self.y, name=name, *args, **kwargs)

except (ImportError, ModuleNotFoundError) as e:
    # Create empty class
    class TH2D:
        def __init__(self, *args):
            raise NotImplementedError(
                "Install ROOT with pyroot enabled to use this feature.")

    class TH1D:
        def __init__(self, *args):
            raise NotImplementedError(
                "Install ROOT with pyroot enabled to use this feature.")
