from tf_agents.environments import tf_py_environment, gym_wrapper
from gym import spaces
from gym import Env
import numpy as np

MIN_REWARD = -200
MAX_REWARD = 200

NUM_STATIC_FEATURES = 18
NUM_DYNAMIC_FEATURES = 1
NUM_FEATURES = NUM_STATIC_FEATURES + NUM_DYNAMIC_FEATURES

DISCRETE = False

class DummySpecEnv(Env):
    """A dummy environment that defines the specs."""
    metadata = {'render.modes': ['human']}

    def __init__(self, num_stocks=1):
        super(DummySpecEnv, self).__init__()

        """
        Source of truth of observation and action spec.
        """
        self.continuous = not DISCRETE
        if DISCRETE:
            self.action_space = spaces.Box(low=np.array(0), high=np.array(4), dtype=np.int32)
        else:
            if num_stocks == 1:
                self.action_space = spaces.Box(low=np.array(-1), high=np.array(1), dtype=np.float32)
            else:
                self.action_space = spaces.Box(low=-np.ones(num_stocks + 1), high=np.ones(num_stocks + 1), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(num_stocks * NUM_FEATURES,), dtype=np.float32)
        self.reward_range = (MIN_REWARD, MAX_REWARD)

    def step(self, action):
        return [], 0, True, {}

    def reset(self):
        return []

    def render(self, mode='human', close=False):
        return []


def get_reward_range():
    return DummySpecEnv.reward_range

def _get_tf_env(num_stocks=1):
    py_env = gym_wrapper.GymWrapper(DummySpecEnv(num_stocks))
    return tf_py_environment.TFPyEnvironment(py_env)

"""
EXPORTED
"""
def get_action_spec(num_stocks=1):
    return _get_tf_env(num_stocks).action_spec()

def get_observation_spec(num_stocks=1):
    return _get_tf_env(num_stocks).observation_spec()

def get_time_step_spec(num_stocks=1):
    return _get_tf_env(num_stocks).time_step_spec()
