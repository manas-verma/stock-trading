from tf_agents.networks import q_network
from tf_agents.specs import BoundedArraySpec
from tf_agents.utils.common import element_wise_squared_loss

from tf_agents.agents.dqn.dqn_agent import DqnAgent, DdqnAgent
from tf_agents.agents.ddpg.ddpg_agent import DdpgAgent
from tf_agents.agents.sac.sac_agent import SacAgent
from tf_agents.agents.ddpg import actor_network
from tf_agents.agents.ddpg import critic_network
from tf_agents.agents.ddpg import actor_rnn_network
from tf_agents.agents.ddpg import critic_rnn_network
from tf_agents.trajectories import time_step as tf_time_step
from tf_agents.policies import policy_saver

from tf_agents.networks import actor_distribution_network, normal_projection_network, network


from tf_agents.trajectories.time_step import TimeStep, StepType, time_step_spec
from tf_agents.environments import py_environment
import tensorflow as tf
import numpy as np

from keras.models import load_model

from feature_generator import FeatureGenerator
import spec

USE_RNN = False
USE_DDPG = False
USE_SAC = True

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
# RNN
CRITIC_LSTM_SIZE = (10,)
CRITIC_OUTPUT_FC_LAYERS = (10,)

ALPHA_ALPHA = DEFAULT_ALPHA * 2


class Agent:

    def __init__(self, ticker):
        self.ticker = ticker
        if USE_DDPG:
            self.actor = self.create_actor_network()
            self.critic = self.create_critic_network()
            self.agent = self.create_ddpg_agent(self.actor, self.critic)
        elif USE_SAC:
            self.actor = self.create_actor_network()
            self.critic = self.create_critic_network()
            self.agent = self.create_sac_agent(self.actor, self.critic)
        else:
            self.q_network = self.create_q_network()
            self.agent = self.create_dqn_agent(self.q_network)

        self.agent.initialize()

    def create_actor_network(self):
        if USE_RNN:
            return actor_rnn_network.ActorRnnNetwork(
                spec.get_observation_spec(),
                spec.get_action_spec(),
                input_fc_layer_params=ACTOR_FC_LAYERS,
                lstm_size=ACTOR_LSTM_LAYERS,
                output_fc_layer_params=ACTOR_OUTPUT_FC_LAYERS)
        else:
            return actor_network.ActorNetwork(
                    spec.get_observation_spec(),
                    spec.get_action_spec(),
                    fc_layer_params=ACTOR_FC_LAYERS,
                    dropout_layer_params=ACTOR_DROPOUT_LAYER_PARAMS,
                    name='actor_' + self.ticker)

    def create_critic_network(self):
        critic_net_input_specs = (spec.get_observation_spec(),
                                spec.get_action_spec())
        if USE_RNN:
            return critic_rnn_network.CriticRnnNetwork(
                critic_net_input_specs,
                observation_fc_layer_params=CRITIC_OBS_FC_LAYERS,
                action_fc_layer_params=CRITIC_ACTION_FC_LAYERS,
                joint_fc_layer_params=CRITIC_JOINT_FC_LAYERS,
                lstm_size=CRITIC_LSTM_SIZE,
                output_fc_layer_params=CRITIC_OUTPUT_FC_LAYERS)
        else:
            return critic_network.CriticNetwork(
                    critic_net_input_specs,
                    observation_fc_layer_params=CRITIC_OBS_FC_LAYERS,
                    action_fc_layer_params=CRITIC_ACTION_FC_LAYERS,
                    joint_fc_layer_params=CRITIC_JOINT_FC_LAYERS,
                    name='critic_' + self.ticker)

    def create_sac_agent(self, actor, critic):
        train_step_counter = tf.Variable(0)
        return SacAgent(
                spec.get_time_step_spec(),
                spec.get_action_spec(),
                actor_network=actor,
                critic_network=critic,
                actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=ACTOR_ALPHA),
                critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=CRITIC_ALPHA),
                alpha_optimizer=tf.compat.v1.train.AdamOptimizer(
                    learning_rate=ALPHA_ALPHA),
                target_update_tau=0.05,
                target_update_period=5,
                td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
                gamma=GAMMA,
                train_step_counter=train_step_counter)

    def create_ddpg_agent(self, actor, critic):
        return DdpgAgent(spec.get_time_step_spec(),
                                    spec.get_action_spec(),
                                    actor_network=actor,
                                    critic_network=critic,
                                    actor_optimizer=tf.compat.v1.train.AdamOptimizer(
                                        learning_rate=ACTOR_ALPHA),
                                    critic_optimizer=tf.compat.v1.train.AdamOptimizer(
                                        learning_rate=CRITIC_ALPHA),
                                    ou_stddev=0.2,
                                    ou_damping=0.15,
                                    target_update_tau=0.05,
                                    target_update_period=5,
                                    td_errors_loss_fn=tf.compat.v1.losses.huber_loss,
                                    gamma=GAMMA,)

    def create_q_network(self):
        return q_network.QNetwork(spec.get_observation_spec(),
                                          spec.get_action_spec(),
                                          fc_layer_params=Q_NET_FC_LAYER_PARAMS)

    def create_dqn_agent(self, q_network):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=Q_ALPHA)
        train_step_counter = tf.Variable(0)
        return DdqnAgent(spec.get_time_step_spec(),
                              spec.get_action_spec(),
                              q_network=q_network,
                              optimizer=optimizer,
                              td_errors_loss_fn=element_wise_squared_loss,
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
