from ..feature_generator import FeatureGenerator
from ..feature_sets import SimpleFibFeatureSet
from ..agent_base import AgentBase


THRESHOLD = 0.2


class WilliamsAlligator(AgentBase):

    def __init__(self, name, stock_list, threshold=THRESHOLD):
        self.name = name
        self.feature_generator = FeatureGenerator((SimpleFibFeatureSet,))
        self.stock_list = stock_list
        self.threshold = threshold
        self.reset()

    def reset(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        self.feature_generator.reset()
        for stock in stock_list:
            self.feature_generator.append_initial_prices(stock.get_prices(1000))

    def take_action(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        action_set = dict()
        for stock in stock_list:
            action = self.take_action_for_stock(stock)
            action_set[stock.ticker] = action / len(stock_list)

        return action_set

    def take_action_for_stock(self, stock):
        self.feature_generator.append_price(stock.price())
        features = self.feature_generator.get_features()
        fib_vals = list(features[:3][::-1])
        fib_ema_slow, fib_ema_mid, fib_ema_fast = fib_vals

        # Sleeping
        if max(fib_vals) - min(fib_vals) < self.threshold:
            return 0.5

        # Trending up and down.
        if sorted(fib_vals) == fib_vals:
            return 0.9
        if sorted(fib_vals[::-1]) == fib_vals[::-1]:
            return 0.0

        # Sharp turn up and down.
        if fib_ema_mid < fib_ema_slow <= fib_ema_fast:
            return 0.75
        if fib_ema_fast < fib_ema_slow <= fib_ema_mid:
            return 0.2

        if fib_ema_slow <= fib_ema_mid:
            return 0.8
        else:
            return 0.1


