from tf_agents.networks import q_network
from tf_agents.specs import BoundedArraySpec
from tf_agents.utils.common import element_wise_huber_loss

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
from ..feature_sets.ema_feature_set import EmaFeatureSet
from evaluator import EvalEnv
import util

ACTION_VALUES = [0.0, 0.1, 0.5, 0.75, 0.9]
OVERALL_ALPHA = 0.75

INIT_TEMP = 20.0
COOLDOWN_TIME = 3
EPSILON = None

"""
Input the following into wolframalpha.com to see the discounted values for an hour, day, two days, and week.
{(1.0 - a)^(60.0), (1.0 - a)^(390.0), (1.0 - a)^(2.0 * 390.0), (1.0 - a)^(7.0 * 390.0)}, where a=4.5e-4
"""
GAMMA = 1 - 4.5e-4


DEFAULT_ALPHA = 1e-7

Q_ALPHA = DEFAULT_ALPHA
Q_NET_FC_LAYER_PARAMS = (64,16)

AVG_STEPS_PER_EPISODE = 390
BATCH_SIZE = 4

INIT_COLLECT_STEPS = 10 * AVG_STEPS_PER_EPISODE
STEP_ITERATIONS = AVG_STEPS_PER_EPISODE
LOG_INTERVAL = 5 * STEP_ITERATIONS

MAX_BUFFER_SIZE = 100000
NUM_EVAL_EPISODES = 100
EVAL_INTERVAL = 2 * LOG_INTERVAL

TRAINING_ITERATIONS = 50

EMA_WINDOW = 390


class DdqnAgent(AgentBase):

    def __init__(self, name, stock_list, feature_set_factories,
                                         action_values=ACTION_VALUES,
                                         overall_alpha=OVERALL_ALPHA,
                                         fc_layer_params=Q_NET_FC_LAYER_PARAMS,
                                         dropout_layer_params=None,
                                         alpha=Q_ALPHA,
                                         gamma=GAMMA,
                                         epsilon=EPSILON,
                                         init_temp=INIT_TEMP,
                                         cooldown_time=COOLDOWN_TIME,
                                         use_rolling_bands=True):
        tf.random.set_seed(0)
        self.name = name
        self.convert_action = self._convert_action_rolling if use_rolling_bands else self._convert_action_simple

        self.stock_list = stock_list
        self.global_step_val = 0
        self.action_values = action_values
        self.overall_alpha = overall_alpha

        self.get_feature_generator = lambda: FeatureGenerator(feature_set_factories)

        self.reset()
        action_space = self.action_space = spaces.Box(low=0,
                                                      high=len(self.action_values) - 1,
                                                      shape=(1,),
                                                      dtype=np.float32)
        self.gym_training_env = TrainingStockEnv(stock_list, self.get_feature_generator(), action_space, self)
        self.tf_training_env = tf_py_environment.TFPyEnvironment(
                gym_wrapper.GymWrapper(
                        self.gym_training_env,
                        discount=gamma,
                        auto_reset=True))

        self.q_network = self.create_q_network(fc_layer_params, dropout_layer_params)
        self.tf_agent = self.create_tf_ddqn_agent(
                                self.q_network,
                                alpha,
                                gamma,
                                epsilon,
                                init_temp,
                                cooldown_time)
        self.eval_policy = self.tf_agent.policy
        self.eval_env = EvalEnv(self.stock_list, self)

        self.tf_agent.initialize()

    def create_q_network(self, fc_layer_params, dropout_layer_params):
        return q_network.QNetwork(self.tf_training_env.observation_spec(),
                                  self.tf_training_env.action_spec(),
                                  fc_layer_params=fc_layer_params,
                                  dropout_layer_params=dropout_layer_params)

    def create_tf_ddqn_agent(self, q_network, alpha, gamma, epsilon, init_temp, cooldown_time):
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=alpha)
        train_step_counter = tf.Variable(0)
        _temp = tf.Variable(np.float(init_temp)) - tf.dtypes.cast(train_step_counter, tf.float32) * tf.Variable((init_temp - 1) / (STEP_ITERATIONS * cooldown_time))

        if epsilon is not None:
            _temp = None

        return dqn_agent.DdqnAgent(self.tf_training_env.time_step_spec(),
                         self.tf_training_env.action_spec(),
                         q_network=q_network,
                         optimizer=optimizer,
                         gamma=gamma,
                         epsilon_greedy=epsilon,
                         boltzmann_temperature=_temp,
                         td_errors_loss_fn=element_wise_huber_loss,
                         train_step_counter=train_step_counter,
                         gradient_clipping=10.0)

    def load(self):
        try:
            self.eval_policy = tf.saved_model.load('model_data/' + self.name)
        except Exception as e:
            print("Caught Error:", e)
            self.eval_policy = self.tf_agent.policy

    def save(self):
        policy_saver.PolicySaver(self.tf_agent.policy).save('model_data/' + self.name)
        self.eval_policy = self.tf_agent.policy

    def reset(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        self.feature_generators = dict()
        self.ema_feature_sets = dict()
        for stock in stock_list:
            self.feature_generators[stock.ticker] = self.get_feature_generator()
            self.ema_feature_sets[stock.ticker] = EmaFeatureSet((EMA_WINDOW, ))

        for stock in stock_list:
            prices = stock.get_prices(1000)
            feature_generator = self.feature_generators[stock.ticker]
            feature_set = self.ema_feature_sets[stock.ticker]

            feature_generator.reset()
            feature_generator.append_initial_prices(prices)
            feature_set.append_initial_prices(prices)

    def take_action(self, eval_stock_list=None):
        """ Returns percentage of equity to invest.
        """
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        action_set = dict()
        for stock in stock_list:
            action = self.take_action_for_stock(stock)
            action_set[stock.ticker] = OVERALL_ALPHA * action / len(stock_list)

        return action_set

    def take_action_for_stock(self, stock):
        feature_generator = self.feature_generators[stock.ticker]
        feature_set = self.ema_feature_sets[stock.ticker]

        price = stock.price()
        feature_generator.append_price(price)
        feature_set.append_price(price)

        observation = feature_generator.get_features()
        time_step = TimeStep(
                step_type=np.array([1], dtype=np.int32),
                reward=np.array([1.0], dtype=np.float32),
                discount=np.array([1.001], dtype=np.float32),
                observation=np.array([observation], dtype=np.float32))
        action = self.convert_action(self, self.eval_policy.action(time_step)[0], stock.ticker, price)
        return action

    def _convert_action_simple(self, agent, action, ticker, price):
        return agent.action_values[int(action)]

    def _convert_action_rolling(self, agent, action, ticker, price):
        feature_set = agent.ema_feature_sets[ticker]
        mu = feature_set.mean[0]
        var = feature_set.var[0] * agent.action_values[int(action)]

        N = 1000
        alpha = 0.5
        for _ in range(N):
            M = 1 + mu * alpha
            V = alpha * alpha * var
            alpha = mu * (M * M + V) / (var * M)
        return max(0, min(1, alpha))

    def compute_avg_return(self, policy):
        avg_return = 0.0
        avg_return_per_step = 0.0

        try:
            for _ in range(NUM_EVAL_EPISODES):

                time_step = self.tf_training_env.reset()
                episode_return = 0.0


                num_steps = 0
                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = self.tf_training_env.step(action_step.action)
                    episode_return += time_step.reward
                    num_steps += 1
                avg_return += episode_return
                if num_steps > 0:
                    avg_return_per_step += episode_return / num_steps

            avg_return /= NUM_EVAL_EPISODES
            avg_return_per_step /= NUM_EVAL_EPISODES
            avg_return_per_step_val = avg_return_per_step.numpy()[0]
            return avg_return.numpy()[0], avg_return_per_step_val, 100 * (np.exp(avg_return_per_step_val * 390) - 1)
        except Exception as e:
            print("Caught Exception:", e)
            return -100, -100, -100

    def train(self, training_iterations=TRAINING_ITERATIONS, training_stock_list=None):
        self.reset(training_stock_list)

        train_dir = 'training_data_progress/train-' + self.name
        eval_dir = 'training_data_progress/eval-' + self.name

        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                    data_spec=self.tf_agent.collect_data_spec,
                    batch_size=self.tf_training_env.batch_size,
                    max_length=MAX_BUFFER_SIZE)

        summaries_flush_secs = 10

        eval_metrics = [
                tf_metrics.AverageReturnMetric(buffer_size=NUM_EVAL_EPISODES),
                tf_metrics.AverageEpisodeLengthMetric(buffer_size=NUM_EVAL_EPISODES)
        ]

        global_step = self.tf_agent.train_step_counter
        with tf.compat.v2.summary.record_if(
                lambda: tf.math.equal(global_step % LOG_INTERVAL, 0)):

            replay_observer = [replay_buffer.add_batch]

            train_metrics = [
                    tf_metrics.NumberOfEpisodes(),
                    tf_metrics.EnvironmentSteps(),
                    tf_metrics.AverageReturnMetric(
                            buffer_size=NUM_EVAL_EPISODES, batch_size=self.tf_training_env.batch_size),
                    tf_metrics.AverageEpisodeLengthMetric(
                            buffer_size=NUM_EVAL_EPISODES, batch_size=self.tf_training_env.batch_size),
            ]

            eval_policy = greedy_policy.GreedyPolicy(self.tf_agent.policy)
            initial_collect_policy = random_tf_policy.RandomTFPolicy(
                    self.tf_training_env.time_step_spec(), self.tf_training_env.action_spec())
            collect_policy = self.tf_agent.collect_policy

            train_checkpointer = common.Checkpointer(
                    ckpt_dir=train_dir,
                    agent=self.tf_agent,
                    global_step=global_step,
                    metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics'))
            policy_checkpointer = common.Checkpointer(
                    ckpt_dir=os.path.join(train_dir, 'policy'),
                    policy=eval_policy,
                    global_step=global_step)
            rb_checkpointer = common.Checkpointer(
                    ckpt_dir=os.path.join(train_dir, 'replay_buffer'),
                    max_to_keep=1,
                    replay_buffer=replay_buffer)

            train_checkpointer.initialize_or_restore()
            rb_checkpointer.initialize_or_restore()

            initial_collect_driver_random = dynamic_step_driver.DynamicStepDriver(
                    self.tf_training_env,
                    initial_collect_policy,
                    observers=replay_observer + train_metrics,
                    num_steps=INIT_COLLECT_STEPS)
            initial_collect_driver_random.run = common.function(initial_collect_driver_random.run)

            collect_driver = dynamic_step_driver.DynamicStepDriver(
                    self.tf_training_env,
                    collect_policy,
                    observers=replay_observer + train_metrics,
                    num_steps=STEP_ITERATIONS)

            collect_driver.run = common.function(collect_driver.run)
            self.tf_agent.train = common.function(self.tf_agent.train)


            # Collect some initial data.
            # Random
            random_policy = random_tf_policy.RandomTFPolicy(self.tf_training_env.time_step_spec(), self.tf_training_env.action_spec())
            avg_return, avg_return_per_step, avg_daily_percentage = self.compute_avg_return(random_policy)
            print('Random:\n\tAverage Return = {0}\n\tAverage Return Per Step = {1}\n\tPercent = {2}%'.format(avg_return, avg_return_per_step, avg_daily_percentage))
            self.gym_training_env.save_feature_distribution(self.name)

            # Agent
            avg_return, avg_return_per_step, avg_daily_percentage = self.compute_avg_return(self.tf_agent.policy)
            print('Agent :\n\tAverage Return = {0}\n\tAverage Return Per Step = {1}\n\tPercent = {2}%'.format(avg_return, avg_return_per_step, avg_daily_percentage))
            self.eval_env.reset()
            self.eval_env.run_and_save_evaluation(str(0))
            self.gym_training_env.save_feature_distribution(self.name)

            evaluations = [self.get_evaluation()]
            returns = [self.eval_env.returns]
            actions_over_time_list = [self.eval_env.action_sets_over_time]

            # Collect initial replay data.
            print(
                    'Initializing replay buffer by collecting experience for {} steps with '
                    'a random policy.'.format(INIT_COLLECT_STEPS))
            initial_collect_driver_random.run()

            results = metric_utils.eager_compute(
                    eval_metrics,
                    self.tf_training_env,
                    eval_policy,
                    num_episodes=NUM_EVAL_EPISODES,
                    train_step=global_step,
                    summary_prefix='Metrics',
            )
            metric_utils.log_metrics(eval_metrics)

            time_step = None
            policy_state = collect_policy.get_initial_state(self.tf_training_env.batch_size)

            timed_at_step = global_step.numpy()
            time_acc = 0

            # Prepare replay buffer as dataset with invalid transitions filtered.
            def _filter_invalid_transition(trajectories, unused_arg1):
                return ~trajectories.is_boundary()[0]
            dataset = replay_buffer.as_dataset(
                    sample_batch_size=BATCH_SIZE,
                    num_steps=2).unbatch().filter(
                            _filter_invalid_transition).batch(BATCH_SIZE).prefetch(5)
            # Dataset generates trajectories with shape [Bx2x...]
            iterator = iter(dataset)

            def _train_step():
                try:
                    experience, _ = next(iterator)
                    return self.tf_agent.train(experience)
                except Exception as e:
                    print("Caught Exception:", e)
                    return 1e-20

            train_step = common.function(_train_step)

            for _ in range(training_iterations):
                start_time = time.time()
                time_step, policy_state = collect_driver.run(
                        time_step=time_step,
                        policy_state=policy_state,
                )
                for _ in range(STEP_ITERATIONS):
                    train_loss = train_step()
                time_acc += time.time() - start_time

                self.global_step_val = global_step.numpy()

                if self.global_step_val % LOG_INTERVAL == 0:
                    steps_per_sec = (self.global_step_val - timed_at_step) / time_acc
                    print(self.name, '\nstep = {0:d}:\n\tloss = {1:f}\n\t{2:.3f} steps/sec'.format(self.global_step_val, train_loss.loss, steps_per_sec))
                    tf.compat.v2.summary.scalar(
                            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                    timed_at_step = self.global_step_val
                    time_acc = 0

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                            train_step=global_step, step_metrics=train_metrics[:2])

                if self.global_step_val % EVAL_INTERVAL == 0:
                    results = metric_utils.eager_compute(
                            eval_metrics,
                            self.tf_training_env,
                            eval_policy,
                            num_episodes=NUM_EVAL_EPISODES,
                            train_step=global_step,
                            summary_prefix='Metrics',
                    )
                    metric_utils.log_metrics(eval_metrics)

                    avg_return, avg_return_per_step, avg_daily_percentage = self.compute_avg_return(self.tf_agent.policy)
                    print(self.name, '\nstep = {0}:\n\tloss = {1}\n\tAverage Return = {2}\n\tAverage Return Per Step = {3}\n\tPercent = {4}%'.format(self.global_step_val, train_loss.loss, avg_return, avg_return_per_step, avg_daily_percentage))
                    self.eval_env.reset()
                    self.eval_env.run_and_save_evaluation(str(self.global_step_val // EVAL_INTERVAL))
                    self.gym_training_env.save_feature_distribution(self.name)

                    if avg_daily_percentage == returns[-1]:
                        "---- Average return did not change since last time. Breaking loop."
                        break

                    evaluations.append(self.get_evaluation())
                    returns.append(self.eval_env.returns)
                    actions_over_time_list.append(self.eval_env.action_sets_over_time)

                    train_checkpointer.save(global_step=self.global_step_val)
                    policy_checkpointer.save(global_step=self.global_step_val)
                    rb_checkpointer.save(global_step=self.global_step_val)

        training_report = util.load_training_report()
        agent_report = training_report.get(self.name, dict())
        agent_report["Training Results"] = returns
        agent_report["Evaluations"] = [max(e, 0.0) for e in evaluations]
        bins = [0.1 * i - 0.0000001 for i in range(11)]
        agent_report["Histograms"] = [str(list(map(int, np.histogram(actions, bins, density=True)[0]))) for actions in actions_over_time_list]
        training_report[self.name] = agent_report
        util.save_training_report(training_report)

        print("---- Average-daily-percentage over training period for", self.name)
        print("\t\t", avg_daily_percentage)
        self.save()
        self.reset()