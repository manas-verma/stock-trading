from ..agent_base import AgentBase
from ..feature_sets import EmaFeatureSet

import numpy as np


NUM_ITER = 12
WINDOW = 390

class MaxLog(AgentBase):

    def __init__(self, name, stock_list, num_iter=NUM_ITER, window=WINDOW):
        self.name = name
        self.stock_list = stock_list
        self.num_iter = num_iter
        self.window = window
        self.reset()

    def reset(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        self.feature_sets = {stock.ticker: EmaFeatureSet((self.window,)) for stock in stock_list}
        for stock in stock_list:
            self.feature_sets[stock.ticker].append_initial_prices(stock.get_prices(390))

    def get_single_alpha(self, mu, var):
        N = 1000
        alpha = 0.5
        for _ in range(N):
            M = 1 + mu * alpha
            V = alpha * alpha * var
            alpha = mu * (M * M + V) / (var * M)
        return max(0, min(1, alpha))

    def take_action(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        N = len(stock_list)

        alpha = np.ones(N) / (N + 1)
        mean = []
        var = []

        for stock in stock_list:
            price = stock.price()
            feature_set = self.feature_sets[stock.ticker]
            feature_set.append_price(price)

            mean.append(feature_set.mean[0])
            var.append(feature_set.var[0])

        mean = np.array(mean)
        var = np.array(var)

        for _ in range(self.num_iter):
            M = 1 + (alpha * mean).sum()
            V = (alpha * alpha * var).sum()

            alpha = mean * (M * M + V) / (var * M)

            for i in range(len(alpha)):
                if alpha[i] < 0.0:
                    alpha[i] = 0.0
                if alpha[i] > 1.0:
                    alpha[i] = self.get_single_alpha(mean[i], var[i])
            if alpha.sum() > 1.0:
                alpha /= alpha.sum()

        action_set = dict()
        for action, stock in zip(alpha, stock_list):
            action_set[stock.ticker] = action

        return action_set
