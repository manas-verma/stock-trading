import numpy as np


class FeatureGenerator:

    def __init__(self, feature_set_factories):
        self.feature_set_factories = feature_set_factories
        self.reset()

    def copy(self):
        return FeatureGenerator(self.feature_set_factories)

    def reset(self):
        self.feature_sets = [feature_set_factory() for feature_set_factory in self.feature_set_factories]

    def append_initial_prices(self, prices):
        for feature_set in self.feature_sets:
            feature_set.append_initial_prices(prices)

    def append_price(self, price):
        for feature_set in self.feature_sets:
            feature_set.append_price(price)

    def __len__(self):
        return sum([len(feature_set) for feature_set in self.feature_sets])

    def feature_spec(self):
        return len(self)

    def get_features(self, dynamic_features=None):
        if dynamic_features is None:
            dynamic_features = []

        static_features = []
        for feature_set in self.feature_sets:
            static_features.extend(feature_set.get_features())

        features = np.append(static_features, dynamic_features)
        return features.astype(np.float32)

    def get_min_init_price_length(self):
        return max([feature_set.get_min_init_price_length() for feature_set in self.feature_sets])
