from . import FeatureSetBase

from hurst import compute_Hc

class HurstFeatureSet(FeatureSetBase):

    def __init__(self, windows=(100,)):
        self.windows = windows
        self.prices = []

    def append_initial_prices(self, prices):
        self.prices = prices

    def __len__(self):
        return len(self.windows)

    def append_price(self, price):
        self.prices.pop(0)
        self.prices.append(price)

    def get_features(self):
        hursts = []
        for window in self.windows:
            H, _, _ = compute_Hc(self.prices[-window:], kind='price', simplified=True)
            hursts.append(H)

        return np.array(hursts)

    def get_min_init_price_length(self):
        return max(self.windows)