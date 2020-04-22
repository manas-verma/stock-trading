import numpy as np

class MovingAvg:

    def __init__(self, prices, window):
        self.prices = prices[-window:]
        self.avg = np.avg(self.prices)
        self.std = np.std(self.prices)

    def append_price(self, price):
        self.prices.pop(0)
        self.prices.append(price)
        self.avg = np.avg(self.prices)
        self.std = np.std(self.prices)

    def current_normalized_price(self):
        return (self.price - self.mean) / self.std