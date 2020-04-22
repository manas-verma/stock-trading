import numpy as np
from hurst import compute_Hc

from .exp_moving_avg import ExpMovingAvg

MIN_INIT_PRICE_LENGTH = 200

MOVING_AVERAGE_WINDOWS = [3, 5, 8, 13, 21, 34, 55]


class FeatureGenerator:

    def __init__(self, prices):
        self.prices = prices

        self.overall_ema = ExpMovingAvg(self.prices, len(self.prices))

        self.fib_emas = [ExpMovingAvg(self.prices, window) for window in MOVING_AVERAGE_WINDOWS]

    def append_price(self, price):
        self.prices = np.append(self.prices, [price])
        self.overall_ema.append_price(price)
        for fib_ema in self.fib_emas:
            fib_ema.append_price(price)

    def get_features(self, dynamic_features=None):
        if dynamic_features is None:
            dynamic_features = []

        last_five_prices = [self.overall_ema.normalize_value(p) for p in self.prices[-5:]]
        one_step_price_diff = self.overall_ema.current_normalized_price_diff()
        twenty_step_price_diff = self.overall_ema.normalize_diff(self.prices[-1] - self.prices[-20])
        sixty_step_price_diff = self.overall_ema.normalize_diff(self.prices[-1] - self.prices[-60])
        fib_ema_vals = [self.overall_ema.normalize_value(fib_ema.mean) for fib_ema in self.fib_emas]

        sixty_step_max_price = self.overall_ema.normalize_value(max(self.prices[-60:]))
        sixty_step_min_price = self.overall_ema.normalize_value(min(self.prices[-60:]))

        H, _, _ = compute_Hc(self.prices[-100:], kind='price', simplified=True)

        static_features = last_five_prices
        static_features += [
            one_step_price_diff,
            twenty_step_price_diff,
            sixty_step_price_diff,
            sixty_step_max_price,
            sixty_step_min_price,
            H]
        static_features.extend(fib_ema_vals)

        features = np.append(static_features, dynamic_features)
        return features.astype(np.float32)

    @classmethod
    def get_min_init_price_length(self):
        return MIN_INIT_PRICE_LENGTH

    @classmethod
    def update_dynamic_features(self, feature, new_dynamic_features):
        N = len(feature) - len(new_dynamic_features)
        return feature[:N] + new_dynamic_features
