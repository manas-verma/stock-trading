import numpy as np
from gym import Env, spaces
import random
import matplotlib.pyplot as plt

INIT_EQUITY = 100000
MIN_LENGTH = 390
NOISE_FACTOR = 0.005


class TrainingStockEnv(Env):

    def __init__(self, stock_list, feature_generator, action_space, gym_agent, continuous=False):
        super(TrainingStockEnv, self).__init__()

        self.stock_list = stock_list
        self.feature_generator = feature_generator
        self.continuous = continuous
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(feature_generator.feature_spec(),),
                                            dtype=np.float32)
        self.action_space = action_space
        self.gym_agent = gym_agent
        self.reward_range = (-np.inf, 20)

        self.stock_prices = {stock.ticker: stock.get_training_dataset() for stock in self.stock_list}
        self.init_equity = INIT_EQUITY
        self.equity = INIT_EQUITY

        self.features_over_time = []

        self.reset()

    def _next_observation(self):
        features = self.feature_generator.get_features()

        self.features_over_time.extend(features)

        return self.feature_generator.get_features()

    def step(self, action):
        price = self.prices[self.index]
        price *= np.random.uniform(1 - NOISE_FACTOR, 1 + NOISE_FACTOR)
        self.feature_generator.append_price(price)
        self.index += 1
        done = self.index == self.end_index or self.index == len(self.prices) - 2

        action = self.gym_agent.convert_action(self.gym_agent, action, self.stock_ticker, price)

        future_price_increase = self.prices[self.index + 1] / self.prices[self.index]
        future_equity_increase = 1.0 + (future_price_increase - 1.0) * action
        self.equity *= future_equity_increase
        reward = np.log(future_equity_increase)
        if not (reward >= -np.float('inf')):
            print("Reward is Nan")
            print("prices:", self.prices)
            print("action:", action)
            print("next price:", self.prices[self.index + 1])
            print("curr price:", self.prices[self.index])
            print("equity increase:", 1.0 + (self.prices[self.index + 1] / self.prices[self.index] - 1.0) * action)
            print("ln of equity increase:", np.log(1.0 + (self.prices[self.index + 1] / self.prices[self.index] - 1.0) * action))
            print("future_equity_increase:", future_equity_increase)
            print("future_price_increase:", future_price_increase)
            print("index:", self.index)
            print("randomized price:", price)
            print("old reward:", reward)
            reward = np.log(1.0 + (self.prices[self.index + 1] / self.prices[self.index] - 1.0) * action)
            print("new reward:", reward)

        # Observation must be calculated after reward.
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        self.feature_generator.reset()
        self.stock_ticker = random.choice(list(self.stock_prices.keys()))
        self.prices = np.array(self.stock_prices[self.stock_ticker])

        self.init_equity = INIT_EQUITY
        self.equity = INIT_EQUITY

        S, E = self.feature_generator.get_min_init_price_length(), len(self.prices) - 2
        min_length = min(MIN_LENGTH, E - S - 1)
        self.index = np.random.choice(range(S, E - min_length))
        self.end_index = np.random.choice(range(self.index + min_length, E))

        self.feature_generator.append_initial_prices(self.prices[:self.index])

        obs = self._next_observation()
        reward = 0

        return obs

    def save_feature_distribution(self, agent_name):
        plt.figure(0)

        plt.hist(self.features_over_time)

        plt.tight_layout()
        plt.savefig("training_data_progress/" + agent_name + '-feature-set-hist.png')
        plt.close()