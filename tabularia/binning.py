import numpy as np


class Binning:
    def __init__(self, num_bins=10):
        self.num_bins = num_bins
        self.bins = None
    
    def fit(self, x):
        x_min, x_max = x.min(), x.max()
        self.bins = np.linspace(start=x_min, stop=x_max, num=self.num_bins)
        return self
    
    def transform(self, x):
        x = np.digitize(x, bins=self.bins)
        return x
    
    def fit_transform(self, x):
        encoder = self.fit(x)
        x = encoder.transform(x)
        return x