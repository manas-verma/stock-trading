from .training_stock_env import TrainingStockEnv
import agent
import util

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

BATCH_SIZE = 4
INIT_COLLECT_STEPS = 400
STEP_ITERATIONS = 2000
LOG_INTERVAL = 500
GAMMA = 1 + 1e-1

MAX_BUFFER_SIZE = 100000
NUM_EVAL_EPISODES = 1
EVAL_INTERVAL = 1000


class Trainer:

    def __init__(self, dataset):
        self.stock_gym_envs = {}
        self.stock_train_envs = {}

        self.stock_eval_py_envs = {}
        self.stock_eval_envs = {}

        self.master_agent = agent.Agent("MASTER")
        self.master_agent.load()

        for ticker in dataset:
            print("Loading env for " + ticker)

            prices = dataset[ticker]
            gym_env = self.get_gym_env(ticker, prices)
            eval_gym_env = self.get_gym_env(ticker, prices)
            print("Loaded gym env for " + ticker)

            self.stock_gym_envs[ticker] = gym_env
            self.stock_train_envs[ticker] = self.get_train_env(gym_env)
            print("Loaded train tf env for " + ticker)

            self.stock_eval_py_envs[ticker] = eval_gym_env
            self.stock_eval_envs[ticker] = self.get_train_env(eval_gym_env)
            print("Loaded eval tf env for " + ticker)

    def get_gym_env(self, ticker, prices):
        return TrainingStockEnv(ticker, prices)

    def get_train_env(self, gym_env):
        train_env = tf_py_environment.TFPyEnvironment(
                gym_wrapper.GymWrapper(
                        gym_env,
                        discount=GAMMA,
                        auto_reset=True))
        return train_env


    def train_all(self, training_iterations):
        for ticker in self.stock_gym_envs:
            self.train_stock(ticker, training_iterations)
        # Save evaluations for Master Agent after training is complete.
        self.evaluate_master_agent()

    def evaluate_master_agent(self):
        for ticker in self.stock_gym_envs:
            print("Evaluating Master Agent for {}...".format(ticker))
            eval_gym_env = self.stock_eval_py_envs[ticker]
            eval_env = self.stock_eval_envs[ticker]
            tf_master_agent = self.master_agent.agent
            master_ticker = "Master:{}".format(ticker)
            self.save_final_evaluation(eval_env, eval_gym_env, tf_master_agent, master_ticker)
            print("Done evaluating.")

    def collect_step(self, stock_env, policy, replay_buffer):
        time_step = stock_env.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = stock_env.step(action_step.action)
        traj = trajectory.from_transition(time_step,
                                          action_step,
                                          next_time_step)
        replay_buffer.add_batch(traj)

    def compute_avg_return(self, environment, py_environment, policy, num_episodes=10):
        total_return = 0.0
        avg_return_per_step = 0.0
        py_environment.random = False

        for _ in range(num_episodes):

            time_step = environment.reset()
            episode_return = 0.0

            num_steps = 0
            while not time_step.is_last():
                action_step = policy.action(time_step)
                time_step = environment.step(action_step.action)
                episode_return += time_step.reward
                num_steps += 1
            total_return += episode_return
            avg_return_per_step += episode_return / num_steps

        py_environment.random = True
        avg_return = total_return / num_episodes
        avg_return_per_step /= num_episodes
        return avg_return.numpy()[0], avg_return_per_step.numpy()[0]


    def save_progress_graph(self, ticker, returns, num_iterations, eval_interval):
        iterations = range(0, num_iterations + 1, eval_interval)
        plt.plot(iterations, returns)
        plt.ylabel('Average Return for ' + ticker)
        plt.xlabel('Iterations')
        plt.savefig('training/progress/' + ticker + '_returns.png')


    def train_stock(self, ticker, training_iterations):
        print()
        print("Training model for " + ticker + "...")
        print()

        gym_env = self.stock_gym_envs[ticker]
        train_env = self.stock_train_envs[ticker]
        eval_gym_env = self.stock_eval_py_envs[ticker]
        eval_env = self.stock_eval_envs[ticker]

        tf_agent = gym_env.agent.agent
        tf_master_agent = self.master_agent.agent

        self.train(ticker, gym_env, train_env, eval_gym_env, eval_env, tf_agent, training_iterations)

        print('Saving weights for {}...'.format(ticker))
        gym_env.agent.save()
        print('Done saving weights for {}.'.format(ticker))

        print("Evaluating...")
        self.save_final_evaluation(eval_env, eval_gym_env, tf_agent, ticker)
        print("Done evaluating.")

        master_ticker = "Master:{}".format(ticker)

        print()
        print("Training model for " + master_ticker + "...")
        print()

        self.train(master_ticker, gym_env, train_env, eval_gym_env, eval_env, tf_master_agent, training_iterations)

        print('Saving weights for Master Agent...')
        self.master_agent.save()
        print('Done saving weights for Master Agent.')

    def train(self, ticker, gym_env, train_env, eval_gym_env, eval_env, tf_agent, training_iterations):
        train_dir = 'training_data_progress/train'
        eval_dir = 'training_data_progress/eval'

        replay_buffer = TFUniformReplayBuffer(
                    data_spec=tf_agent.collect_data_spec,
                    batch_size=train_env.batch_size,
                    max_length=MAX_BUFFER_SIZE)

        summaries_flush_secs = 10

        eval_metrics = [
                tf_metrics.AverageReturnMetric(buffer_size=NUM_EVAL_EPISODES),
                tf_metrics.AverageEpisodeLengthMetric(buffer_size=NUM_EVAL_EPISODES)
        ]

        global_step = tf_agent.train_step_counter
        with tf.compat.v2.summary.record_if(
                lambda: tf.math.equal(global_step % LOG_INTERVAL, 0)):

            replay_observer = [replay_buffer.add_batch]

            train_metrics = [
                    tf_metrics.NumberOfEpisodes(),
                    tf_metrics.EnvironmentSteps(),
                    tf_metrics.AverageReturnMetric(
                            buffer_size=NUM_EVAL_EPISODES, batch_size=train_env.batch_size),
                    tf_metrics.AverageEpisodeLengthMetric(
                            buffer_size=NUM_EVAL_EPISODES, batch_size=train_env.batch_size),
            ]

            eval_policy = greedy_policy.GreedyPolicy(tf_agent.policy)
            initial_collect_policy = random_tf_policy.RandomTFPolicy(
                    train_env.time_step_spec(), train_env.action_spec())
            collect_policy = tf_agent.collect_policy

            train_checkpointer = common.Checkpointer(
                    ckpt_dir=train_dir,
                    agent=tf_agent,
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
                    train_env,
                    initial_collect_policy,
                    observers=replay_observer + train_metrics,
                    num_steps=INIT_COLLECT_STEPS)
            initial_collect_driver_random.run = common.function(initial_collect_driver_random.run)

            always_50 = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), BoundedTensorSpec(shape=(), dtype=tf.float32, name='action', minimum=np.array(0.0, dtype=np.float32), maximum=np.array(0.0, dtype=np.float32)))
            initial_collect_driver_always_50 = dynamic_step_driver.DynamicStepDriver(
                    train_env,
                    always_50,
                    observers=replay_observer + train_metrics,
                    num_steps=INIT_COLLECT_STEPS)
            initial_collect_driver_always_50.run = common.function(initial_collect_driver_always_50.run)

            close_to_50 = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), BoundedTensorSpec(shape=(), dtype=tf.float32, name='action', minimum=np.array(-0.2, dtype=np.float32), maximum=np.array(0.2, dtype=np.float32)))
            initial_collect_driver_close_to_50 = dynamic_step_driver.DynamicStepDriver(
                    train_env,
                    always_50,
                    observers=replay_observer + train_metrics,
                    num_steps=INIT_COLLECT_STEPS)
            initial_collect_driver_close_to_50.run = common.function(initial_collect_driver_close_to_50.run)


            collect_driver = dynamic_step_driver.DynamicStepDriver(
                    train_env,
                    collect_policy,
                    observers=replay_observer + train_metrics,
                    num_steps=STEP_ITERATIONS)

            collect_driver.run = common.function(collect_driver.run)
            tf_agent.train = common.function(tf_agent.train)


            # Collect some initial data.
            # Random
            random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), train_env.action_spec())
            avg_return, avg_return_per_step = self.compute_avg_return(eval_env, eval_gym_env, random_policy, NUM_EVAL_EPISODES)
            percent = 100 * (eval_gym_env.reward_generator.equity() / eval_gym_env.reward_generator.init_equity - 1)
            print('Random:\n\tAverage Return = {0}\n\tAverage Return Per Step = {1}\n\tPercent = {2}%'.format(avg_return, avg_return_per_step, percent))

            avg_return, avg_return_per_step = self.compute_avg_return(eval_env, eval_gym_env, always_50, NUM_EVAL_EPISODES)
            percent = 100 * (eval_gym_env.reward_generator.equity() / eval_gym_env.reward_generator.init_equity - 1)
            print('Always 50%:\n\tAverage Return = {0}\n\tAverage Return Per Step = {1}\n\tPercent = {2}%'.format(avg_return, avg_return_per_step, percent))

            always_100 = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(), BoundedTensorSpec(shape=(), dtype=tf.float32, name='action', minimum=np.array(1.0, dtype=np.float32), maximum=np.array(1.0, dtype=np.float32)))
            avg_return, avg_return_per_step = self.compute_avg_return(eval_env, eval_gym_env, always_100, NUM_EVAL_EPISODES)
            percent = 100 * (eval_gym_env.reward_generator.equity() / eval_gym_env.reward_generator.init_equity - 1)
            print('Always 100%:\n\tAverage Return = {0}\n\tAverage Return Per Step = {1}\n\tPercent = {2}%'.format(avg_return, avg_return_per_step, percent))

            # Agent
            avg_return, avg_return_per_step = self.compute_avg_return(eval_env, eval_gym_env, tf_agent.policy, NUM_EVAL_EPISODES)
            percent = 100 * (eval_gym_env.reward_generator.equity() / eval_gym_env.reward_generator.init_equity - 1)
            returns = [avg_return]
            print('Agent:\n\tAverage Return = {0}\n\tAverage Return Per Step = {1}\n\tPercent = {2}%'.format(avg_return, avg_return_per_step, percent))
            eval_gym_env.reward_generator.save_run_fig("training_data_progress/trainstep_FIRST")
            eval_gym_env.reward_generator.save_run_fig("training_data_progress/trainstep_{1}_{0:}".format(str(0).zfill(5), ticker))

            # Collect initial replay data.
            print(
                    'Initializing replay buffer by collecting experience for {} steps with '
                    'a random policy.'.format(INIT_COLLECT_STEPS))
            initial_collect_driver_random.run()
            initial_collect_driver_close_to_50.run()
            initial_collect_driver_always_50.run()

            results = metric_utils.eager_compute(
                    eval_metrics,
                    eval_env,
                    eval_policy,
                    num_episodes=NUM_EVAL_EPISODES,
                    train_step=global_step,
                    summary_prefix='Metrics',
            )
            metric_utils.log_metrics(eval_metrics)

            time_step = None
            policy_state = collect_policy.get_initial_state(train_env.batch_size)

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
                experience, _ = next(iterator)
                return tf_agent.train(experience)

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

                global_step_val = global_step.numpy()

                if global_step_val % LOG_INTERVAL == 0:
                    steps_per_sec = (global_step_val - timed_at_step) / time_acc
                    print(ticker, '\nstep = {0:d}:\n\tloss = {1:f}\n\t{2:.3f} steps/sec'.format(global_step_val, train_loss.loss, steps_per_sec))
                    tf.compat.v2.summary.scalar(
                            name='global_steps_per_sec', data=steps_per_sec, step=global_step)
                    timed_at_step = global_step_val
                    time_acc = 0

                for train_metric in train_metrics:
                    train_metric.tf_summaries(
                            train_step=global_step, step_metrics=train_metrics[:2])

                if global_step_val % EVAL_INTERVAL == 0:
                    results = metric_utils.eager_compute(
                            eval_metrics,
                            eval_env,
                            eval_policy,
                            num_episodes=NUM_EVAL_EPISODES,
                            train_step=global_step,
                            summary_prefix='Metrics',
                    )
                    metric_utils.log_metrics(eval_metrics)

                    avg_return, avg_return_per_step = self.compute_avg_return(eval_env, eval_gym_env, tf_agent.policy, NUM_EVAL_EPISODES)
                    percent = 100 * (eval_gym_env.reward_generator.equity() / eval_gym_env.reward_generator.init_equity - 1)
                    print(ticker, '\nstep = {0}:\n\tloss = {1}\n\tAverage Return = {2}\n\tAverage Return Per Step = {3}\n\tPercent = {4}%'.format(global_step_val, train_loss.loss, avg_return, avg_return_per_step, percent))
                    eval_gym_env.reward_generator.save_run_fig("training_data_progress/trainstep_{1}_{0:}".format(str(global_step_val).zfill(5), ticker))
                    eval_gym_env.reward_generator.save_run_fig("training_data_progress/trainstep_LAST")

                    train_checkpointer.save(global_step=global_step_val)
                    policy_checkpointer.save(global_step=global_step_val)
                    rb_checkpointer.save(global_step=global_step_val)

            return train_loss


    def save_final_evaluation(self, eval_env, eval_gym_env, tf_agent, name):
        evaluations = util.load_agent_evaluations()
        avg_return, avg_return_per_step = self.compute_avg_return(eval_env, eval_gym_env, tf_agent.policy, NUM_EVAL_EPISODES)
        evaluations[name] = eval_gym_env.reward_generator.get_evaluation()
        util.save_agent_evaluations(evaluations)

