from . import FeatureSetBase

import numpy as np


class GrowthTimeSeriesFeatureSet(FeatureSetBase):

    def __init__(self, length):
        self.length = length
        self.growth_time_series = [0.0 for _ in range(self.length)]

    def append_initial_prices(self, prices):
        self.prev_price = prices[0]
        for p in prices[1:]:
            self.append_price(p)

    def append_price(self, price):
        growth = price / self.prev_price - 1
        self.prev_price = price

        self.growth_time_series.pop(0)
        self.growth_time_series.append(growth)

    def get_features(self):
        return np.array(self.growth_time_series)

    def __len__(self):
        return self.length

    def get_min_init_price_length(self):
        return self.length + 1
