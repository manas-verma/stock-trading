from tf_agents.networks import q_network
from tf_agents.specs import BoundedArraySpec
from tf_agents.utils.common import element_wise_squared_loss

from tf_agents.agents.dqn import dqn_agent
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent
from tf_agents.agents.sac import sac_agent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.trajectories import time_step as tf_time_step
from tf_agents.policies import policy_saver
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.metrics import tf_metrics
from tf_agents.utils import common
from tf_agents.eval import metric_utils

from tf_agents.trajectories import trajectory

from tf_agents.environments import tf_py_environment, gym_wrapper
from tf_agents.replay_buffers.tf_uniform_replay_buffer \
    import TFUniformReplayBuffer
from tf_agents.utils import common
from tf_agents.policies import random_tf_policy
from tf_agents.drivers import dynamic_step_driver
from tf_agents.specs import BoundedTensorSpec
import tensorflow as tf

from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_mujoco
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.policies import greedy_policy
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common

import matplotlib.pyplot as plt
import numpy as np
import os
import time

from tf_agents.networks import actor_distribution_network, normal_projection_network, network


from tf_agents.trajectories.time_step import TimeStep, StepType, time_step_spec
from tf_agents.environments import tf_py_environment, gym_wrapper
import tensorflow as tf
import numpy as np

from gym import spaces
from keras.models import load_model

from ..agent_base import AgentBase
from ..training_environment import TrainingStockEnv
from ..feature_generator import FeatureGenerator
from evaluator import EvalEnv


GAMMA = 1 + 1e-1
DEFAULT_ALPHA = 1e-6

Q_ALPHA = DEFAULT_ALPHA
Q_NET_FC_LAYER_PARAMS = (40,8)

ACTOR_ALPHA = DEFAULT_ALPHA
ACTOR_FC_LAYERS = (256,32)
ACTOR_DROPOUT_LAYER_PARAMS = (0.8,0.4)
# RNN
ACTOR_LSTM_LAYERS = (10,)
ACTOR_OUTPUT_FC_LAYERS = (10,8)

CRITIC_ALPHA = DEFAULT_ALPHA * 0.5
CRITIC_OBS_FC_LAYERS = (128,64)
CRITIC_ACTION_FC_LAYERS = (2,)
CRITIC_JOINT_FC_LAYERS = (16,)

ALPHA_ALPHA = DEFAULT_ALPHA * 2


class SacAgent:

    def __init__(self, name,
                       stock_list,
                       feature_set_factories,
                       actor_fc_layers=ACTOR_FC_LAYERS,
                       actor_dropout_layer_params=ACTOR_DROPOUT_LAYER_PARAMS,
                       critic_observation_fc_layer_params=CRITIC_OBS_FC_LAYERS,
                       critic_action_fc_layer_params=CRITIC_ACTION_FC_LAYERS,
                       critic_joint_fc_layer_params=CRITIC_JOINT_FC_LAYERS,
                       actor_alpha=ACTOR_ALPHA,
                       critic_alpha=CRITIC_ALPHA,
                       alpha_alpha=ALPHA_ALPHA,
                       gamma=GAMMA):
        self.name = name
        self.stock_list = stock_list
        self.feature_generator = FeatureGenerator(feature_set_factories)
        self.reset()
        action_space = self.action_space = spaces.Box(low=-1.0,
                                                      high=1.0,
                                                      shape=(1,),
                                                      dtype=np.float32)
        self.gym_training_env = TrainingStockEnv(stock_list, self.feature_generator.copy(), action_space, self.convert_action)
        self.tf_training_env = tf_py_environment.TFPyEnvironment(
                gym_wrapper.GymWrapper(
                        self.gym_training_env,
                        discount=gamma,
                        auto_reset=True))

        self.actor = self.create_actor_network(actor_fc_layers,
                                               actor_dropout_layer_params)
        self.critic = self.create_critic_network(critic_observation_fc_layer_params,
                                                 critic_action_fc_layer_params,
                                                 critic_joint_fc_layer_params)
        self.tf_agent = self.create_sac_agent(self.actor,
                                           self.critic,
                                           actor_alpha,
                                           critic_alpha,
                                           alpha_alpha,
                                           gamma)
        self.eval_policy = self.tf_agent.policy
        self.eval_env = EvalEnv(self.stock_list, self)

        self.tf_agent.initialize()

    def convert_action(self, action):
        return (1.0 + action) / 2.0

    def create_actor_network(self, actor_fc_layers, actor_dropout_layer_params):
        return actor_network.ActorNetwork(
                    spec.get_observation_spec(),
                    spec.get_action_spec(),
                    fc_layer_params=actor_fc_layers,
                    dropout_layer_params=dropout_layer_params,
                    name='actor_' + self.name)

    def create_critic_network(self, observation_fc_layer_params, action_fc_layer_params, joint_fc_layer_params):
        critic_net_input_specs = (spec.get_observation_spec(),
                                spec.get_action_spec())
        return critic_network.CriticNetwork(
                    critic_net_input_specs,
                    observation_fc_layer_params=observation_fc_layer_params,
                    action_fc_layer_params=action_fc_layer_params,
                    joint_fc_layer_params=joint_fc_layer_params,
                    name='critic_' + self.name)

    def create_sac_agent(self, actor, critic, actor_alpha, critic_alpha, alpha_alpha, gamma):
        train_step_counter = tf.Variable(0)
        return sac_agent.SacAgent(
                spec.get_time_step_spec(),
                spec.get_action_spec(),
                actor_network=actor,
                critic_network=critic,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=actor_alpha),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=critic_alpha),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=alpha_alpha),
                target_update_tau=0.05,
                target_update_period=5,
                td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
                gamma=gamma,
                train_step_counter=train_step_counter)

    def load(self):
        print("Initialized new weights for " + self.ticker)
        self.agent.initialize()
        return False

    def save(self):
        policy_saver.PolicySaver(self.agent.policy).save("model_data/" + self.ticker)

    def take_action(self, observation):
        """ Returns percentage of equity to invest. If negative, that is the
        amount to sell.
        """
        reward = 0
        gamma = 0
        step_type = StepType.MID
        time_step = TimeStep(
                step_type=np.array([1], dtype=np.int32),
                reward=np.array([1.0], dtype=np.float32),
                discount=np.array([1.001], dtype=np.float32),
                observation=np.array([observation], dtype=np.float32))
        return float(self.agent.policy.action(time_step).action)
