from ..agent_base import AgentBase

import numpy as np
from hmmlearn import hmm
import pickle
import os

ITER = 100
NUM_COMPONENTS = 5
NOISE_FACTOR = 0.005
GRANULARITY = 0.05

class Hmm(AgentBase):

    def __init__(self, name, stock_list, num_components=NUM_COMPONENTS,
                                         granularity=GRANULARITY,
                                         simple=True):
        self.name = name
        self.stock_list = stock_list
        self.num_components = num_components
        self.granularity = granularity
        self.reset(self.stock_list)

        if simple:
            self.take_action_for_stock = self._simple_take_action_for_stock
        else:
            self.take_action_for_stock = self._complex_take_action_for_stock

    def reset(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        self.gains_per_stock = dict()
        self.last_price_per_stock = dict()
        for stock in stock_list:
            prices = np.array(stock.get_prices(1000))
            gains = prices[1:] / prices[:-1]
            self.last_price_per_stock[stock.ticker] = prices[-1]
            self.gains_per_stock[stock.ticker] = gains

    def load_data(self, stock_list):
        self.data = []
        self.lengths = []
        for stock in stock_list:
            prices = np.array(stock.get_training_dataset())
            gains = prices[1:] / prices[:-1] - 1
            N = len(prices) - 1
            self.gains_per_stock[stock.ticker] = prices

            # Actual prices.
            observations = [[g] for g in gains]
            self.data.extend(observations)
            self.lengths.append(N)

            # Noisy prices.
            prices = np.array([p * np.random.uniform(1 - NOISE_FACTOR, 1 + NOISE_FACTOR) for p in prices])
            gains = prices[1:] / prices[:-1] - 1
            observations = [[g] for g in gains]
            self.data.extend(observations)
            self.lengths.append(N)

            # More noisy prices.
            prices = np.array([p * np.random.uniform(1 - NOISE_FACTOR, 1 + NOISE_FACTOR) for p in prices])
            gains = prices[1:] / prices[:-1] - 1
            observations = [[g] for g in gains]
            self.data.extend(observations)
            self.lengths.append(N)

    def train(self, training_iterations=ITER, training_stock_list=None):
        stock_list = self.stock_list
        if training_stock_list is not None:
            stock_list = training_stock_list

        self.reset(stock_list)
        self.model = hmm.GaussianHMM(n_components=self.num_components, covariance_type="full", n_iter=training_iterations)
        self.load_data(stock_list)
        self.model.fit(self.data, self.lengths)
        self.save()

    def save(self):
        file_path = 'model_data/' + self.name + '/model.pkl'
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)
        with open(file_path, "wb") as file:
            pickle.dump(self.model, file)

    def load(self):
        try:
            with open('model_data/' + self.name + "/model.pkl", "rb") as file:
                self.model = pickle.load(file)
        except Exception as e:
            print("Caught Exception", e)
            self.model = hmm.GaussianHMM(n_components=self.num_components, covariance_type="full", n_iter=ITER)

    def take_action(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        action_set = dict()
        for stock in stock_list:
            price = stock.price()
            self.gains_per_stock[stock.ticker] = self.gains_per_stock.get(stock.ticker, [])
            self.gains_per_stock[stock.ticker] = list(self.gains_per_stock[stock.ticker])
            self.gains_per_stock[stock.ticker].append(price / self.last_price_per_stock[stock.ticker] - 1)
            self.gains_per_stock[stock.ticker] = np.array(self.gains_per_stock[stock.ticker])
            self.last_price_per_stock[stock.ticker] = price

            gains = [[g] for g in self.gains_per_stock[stock.ticker]]
            N = int(1 / self.granularity)
            new_gains = np.array([i * self.granularity for i in range(-N, N + 1)])
            probabilities = np.array([np.exp(self.model.score(gains + [[new_gain]])) for new_gain in new_gains])
            if probabilities.sum() <= 0:
                print("probs <= 0")
                action_set[stock.ticker] = 0
                continue

            probabilities /= probabilities.sum()

            action = self.take_action_for_stock(new_gains, probabilities)
            action_set[stock.ticker] = action / len(stock_list)

        return action_set

    def _simple_take_action_for_stock(self, new_gains, probabilities):
        mean = (new_gains * probabilities).sum()

        return float(int(mean > 0.0))

    def _complex_take_action_for_stock(self, new_gains, probabilities):
        mean = (new_gains * probabilities).sum()
        var = (new_gains * new_gains * probabilities).sum() - mean * mean
        print("mean", mean)
        print("var", var)

        alpha = var - 2 * mean * mean
        det = var * var - 4 * mean * mean * var

        if det < 0:
            print("det < 0")
            return 1.0

        alpha -= np.sqrt(det)
        alpha = alpha / (2 * mean * mean * mean)

        print("alpha:", alpha)
        alpha = max(0.0, min(1.0, alpha))

        return alpha