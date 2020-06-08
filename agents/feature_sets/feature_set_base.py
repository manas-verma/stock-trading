from abc import abstractmethod

class FeatureSetBase:

    def append_initial_prices(self, prices):
        for price in prices:
            self.append_price(price)

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def append_price(self, price):
        pass

    @abstractmethod
    def get_features(self):
        return []

    def get_min_init_price_length(self):
        return 0