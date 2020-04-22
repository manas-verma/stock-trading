from agent import Agent
from feature_generator import FeatureGenerator
from reward_generator import RewardGenerator

from tf_agents.environments import py_environment
from tf_agents.trajectories.time_step import TimeStep, StepType
import numpy as np

from gym import Env
import spec


class TrainingStockEnv(spec.DummySpecEnv):
    """A stock trading environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self, ticker, prices):
        super(TrainingStockEnv, self).__init__()

        self.ticker = ticker
        self.prices = prices

        self.agent = Agent(self.ticker)
        self.agent.load()

        self.random = True
        self.reset()

    def _next_observation(self):
        return self.feature_generator.get_features(self.reward_generator.get_dynamic_features())

    def step(self, action):
        price = self.prices[self.index]
        self.feature_generator.append_price(price)

        self.index += 1
        done = self.index == self.end_index
        reward = self.reward_generator.get_reward(action)

        # Observation must be calculated after reward.
        obs = self._next_observation()

        return obs, reward, done, {}

    def reset(self):
        S, E = FeatureGenerator.get_min_init_price_length(), len(self.prices) - 1
        if not self.random:
            self.index = S
            self.end_index = E
        else:
            self.index = np.random.choice(range(S, E - S // 2 - 3))
            self.end_index = np.random.choice(range(self.index + S // 2, E))

        self.feature_generator = FeatureGenerator(self.prices[:self.index])
        self.reward_generator = RewardGenerator(self.prices, self.index)

        obs = self._next_observation()
        reward = 0

        return obs

    def render(self, mode='human', close=False):
        return self.reward_generator.render()
