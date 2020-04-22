import numpy as np

class ExpMovingAvg:

    def __init__(self, prices, window):
        self.prev_price = 0
        self.price = prices[0]
        self.mean = prices[0]
        self.var = 0
        self.alpha = 2 / (window + 1)
        for p in prices[1:]:
            self.append_price(p)

    def append_price(self, price):
        self.prev_price = self.price
        self.price = price
        diff = price - self.mean
        self.mean += self.alpha * diff
        self.var += self.alpha * diff * diff
        self.var *= (1 - self.alpha)

    def current_normalized_price(self):
        return self.normalize_value(self.price)

    def current_normalized_price_diff(self):
        return self.normalize_diff(self.price - self.prev_price)

    def normalize_value(self, val):
        return (val - self.mean) / np.sqrt(self.var)

    def normalize_diff(self, diff):
        return diff / np.sqrt(self.var)