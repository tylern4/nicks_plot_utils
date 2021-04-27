import numpy as np
from . import Scatter

# Add a fill method that handles array-like for root
try:
    import ROOT

    class TH2D(ROOT.TH2D):
        def __init__(self, *args):
            ROOT.TH2D.__init__(self, *args)

        def fill(self, xs, ys, weight=1):
            if hasattr(xs, "__len__") and hasattr(ys, "__len__"):
                weight = weight if hasattr(
                    weight, "__len__") else np.ones_like(xs)
                for x, y, w in zip(xs, ys, weight):
                    self.Fill(x, y, w)
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
            if hasattr(val, "__len__"):
                weight = weight if hasattr(
                    weight, "__len__") else np.ones_like(val) * weight
                for v, w in zip(val, weight):
                    self.Fill(v, w)
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
            print(
                "ROOT was not found. Install ROOT with pyroot enabled to use this feature.")
            raise NotImplementedError

    class TH1D:
        def __init__(self, *args):
            print(
                "ROOT was not found. Install ROOT with pyroot enabled to use this feature.")
            raise NotImplementedError
