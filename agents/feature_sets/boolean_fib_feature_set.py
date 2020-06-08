from .simple_fib_feature_set import SimpleFibFeatureSet

import numpy as np

class BooleanFibFeatureSet(SimpleFibFeatureSet):

    def get_features(self):
        fib_vals = self.get_normalized_fib_vals()

        increasing = int(fib_vals[2] <= fib_vals[1] <= fib_vals[0])
        decreasing = int(fib_vals[2] > fib_vals[1] > fib_vals[0])

        quick_turn_up = int(fib_vals[1] <= fib_vals[0] < fib_vals[2])
        quick_turn_down = int(fib_vals[1] > fib_vals[0] >= fib_vals[2])

        trending_up = int(fib_vals[1] < fib_vals[2] <= fib_vals[0])
        trending_down = int(fib_vals[1] >= fib_vals[2] > fib_vals[0])

        return np.array([increasing, decreasing,
                         quick_turn_up, quick_turn_down,
                         trending_up, trending_down])

    def __len__(self):
        return 6
