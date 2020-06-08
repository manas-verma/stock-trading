from . import FeatureSetBase
from .ema_feature_set import EmaFeatureSet

import numpy as np

WINDOWS = [5, 8, 13]
OFFSET = [3, 5, 8]
NORMALIZING_WINDOW = 200

class SimpleFibFeatureSet(FeatureSetBase):

    def __init__(self):
        self.length = 3 * len(WINDOWS)
        self.fib_emas = [EmaFeatureSet((window,)) for window in WINDOWS]
        self.normalizer = EmaFeatureSet((NORMALIZING_WINDOW,))

    def append_initial_prices(self, prices):
        self.prices = list(prices)[-max(OFFSET):]

        self.normalizer.append_initial_prices(self.prices)
        for offset, fib_ema in zip(OFFSET, self.fib_emas):
            fib_ema.append_initial_prices(self.prices[:1-offset])

    def append_price(self, price):
        self.prices.append(price)
        self.prices.pop(0)

        self.normalizer.append_price(price)
        for offset, fib_ema in zip(OFFSET, self.fib_emas):
            fib_ema.append_price(self.prices[-offset])

    def get_normalized_fib_vals(self):
        fib_vals = []

        for fib_ema in self.fib_emas:
            fib_vals.extend(
                    self.normalizer.get_normalized_value(
                            fib_ema.get_features()))
        return fib_vals

    def get_features(self):
        fib_vals = self.get_normalized_fib_vals()

        static_features = []

        # Add the regular values.
        static_features.extend(fib_vals)

        # Add combinations of couples.
        for i in range(len(fib_vals)):
            for j in range(i + 1, len(fib_vals)):
                static_features.append(fib_vals[i] - fib_vals[j])

        # Add combinations of triplets.
        for val in fib_vals:
            static_features.append(3 * val - sum(fib_vals))

        return np.array(static_features)

    def __len__(self):
        return self.length

    def get_min_init_price_length(self):
        return max(OFFSET)
