from . import FeatureSetBase
from .ema_feature_set import EmaFeatureSet

import numpy as np

SCALING_FACTOR = 0.1
OVERALL_WINDOW = 390

class MuEmaFeatureSet(FeatureSetBase):

    def __init__(self, moving_windows, overall_window=OVERALL_WINDOW):
        self.length = len(moving_windows)
        self.alpha = 2 / (overall_window + 1)
        self.ema_feature_set = EmaFeatureSet(moving_windows + (overall_window,))

    def append_initial_prices(self, prices):
        self.ema_feature_set.append_initial_prices(prices)

    def append_price(self, price):
        self.ema_feature_set.append_price(price)

    def get_features(self):
        return np.array([
            self.ema_feature_set.get_normalized_value(mu)[-1]
                    for mu in self.ema_feature_set.mean
            ][:-1])

    def __len__(self):
        return self.length

    def get_min_init_price_length(self):
        return 1
