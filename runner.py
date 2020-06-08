import stock
import trader
import util
import agents
import evaluator

import threading
import numpy

EVALUATION_THRESHOLD = -0.5

class Runner:

    def __init__(self):
        self.init_equity = 75000
        self.alpaca = util.get_alpaca()
        self.stock_list = self.load_stock_list(util.get_stock_universe_list())
        self.eval_stock_list = self.load_stock_list(util.get_eval_stock_list())
        self.agent_list = self.create_agent_list(self.stock_list)

    def load_stock_list(self, tickers):
        stock_list = []

        for ticker in tickers:
            try:
                self.alpaca.get_barset(ticker, 'minute', limit=1000)[ticker]
                stock_list.append(stock.Stock(ticker, self.alpaca))
            except Exception as e:
                print("Caught Exception:", e)

        return stock_list


    def create_agent_list(self, stock_list):
        """ TODO: consider using a separate class to do this.
            Essentially, make it such that only one file needs to be changed
            when adding or removing agents.
        """
        ema_feature_set_factory = lambda s: (lambda: agents.feature_sets.EmaFeatureSet(s))
        mu_ema_feature_set_factory = lambda s, w: (lambda: agents.feature_sets.MuEmaFeatureSet(s, w))
        bool_fib_feature_set_factory = agents.feature_sets.BooleanFibFeatureSet
        simple_fib_feature_set_factory = agents.feature_sets.SimpleFibFeatureSet
        time_series_feature_set_factory = lambda i: (lambda: agents.feature_sets.GrowthTimeSeriesFeatureSet(i))


        alligator = agents.williams_alligator.WilliamsAlligator("williams-alligator",
                        stock_list,
                        0.75)

        baseline_all_equity = agents.baseline.Baseline("baseline-1.0", stock_list, alpha=1.0)
        baseline_three_quarters_equity = agents.baseline.Baseline("baseline-0.75", stock_list, alpha=0.75)
        baseline_half_equity = agents.baseline.Baseline("baseline-0.5", stock_list, alpha=0.5)
        baseline_tenth_equity = agents.baseline.Baseline("baseline-0.1", stock_list, alpha=0.1)

        catch_rise = agents.catch_rise.CatchRise(
                        "catch-rise",
                        stock_list)

        rolling_bands_0_1 = agents.rolling_bands.RollingBands(
                        "rolling-bands-0.1",
                        stock_list,
                        deviation_multiplier=0.1)
        rolling_bands_0_5 = agents.rolling_bands.RollingBands(
                        "rolling-bands-0.5",
                        stock_list,
                        deviation_multiplier=0.5)
        rolling_bands_1_0 = agents.rolling_bands.RollingBands(
                        "rolling-bands-1.0",
                        stock_list,
                        deviation_multiplier=1.0)
        rolling_bands_1_5 = agents.rolling_bands.RollingBands(
                        "rolling-bands-1.5",
                        stock_list,
                        deviation_multiplier=1.5)
        rolling_bands_2_0 = agents.rolling_bands.RollingBands(
                        "rolling-bands-2.0",
                        stock_list,
                        deviation_multiplier=2.0)

        rolling_band_action_values = (0.1, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
        alpha_action_values = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)

        max_log_390 = agents.max_log.MaxLog("max-log-390", stock_list, window=390)
        max_log_450 = agents.max_log.MaxLog("max-log-450", stock_list, window=450)
        max_log_585 = agents.max_log.MaxLog("max-log-585", stock_list, window=585)

        deep_agent = agents.ddqn_agent.DdqnAgent(
                        "deep-time-series-alpha-ddqn",
                        stock_list,
                        feature_set_factories=(
                            time_series_feature_set_factory(60),
                            ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)),
                            mu_ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233), 390),
                            bool_fib_feature_set_factory
                        ),
                        action_values=tuple(0.05 * i for i in range(21)),
                        fc_layer_params=(512, 512, 256, 256, 128, 64),
                        dropout_layer_params=(None, 0.8, 0.8, 0.5, 0.5, 0.5),
                        alpha=1e-6,
                        use_rolling_bands=True)

        max_log_deep_agent = agents.max_log_ddqn.MaxLogDdqn(
                        "max-log-deep-time-series-alpha-ddqn",
                        stock_list,
                        feature_set_factories=(
                            time_series_feature_set_factory(60),
                            ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)),
                            mu_ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233), 390),
                            bool_fib_feature_set_factory
                        ),
                        ema_windows=(60, 90, 195, 390, 450, 780),
                        fc_layer_params=(512, 512, 256, 256, 128, 64),
                        dropout_layer_params=(None, 0.8, 0.8, 0.5, 0.5, 0.5),
                        alpha=1e-6)

        mu_ema_rolling_agent = agents.ddqn_agent.DdqnAgent(
                        "mu-ema-rolling-ddqn",
                        stock_list,
                        feature_set_factories=(
                            ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)),
                            mu_ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233), 390),
                        ),
                        action_values=alpha_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=8e-7,
                        use_rolling_bands=True)

        time_and_mu_ema_rolling_agent = agents.ddqn_agent.DdqnAgent(
                        "time-and-mu-ema-rolling-ddqn",
                        stock_list,
                        feature_set_factories=(
                            time_series_feature_set_factory(60),
                            ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)),
                            mu_ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233), 390),
                        ),
                        action_values=alpha_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=8e-7,
                        use_rolling_bands=True)

        time_series_with_rolling_agent = agents.ddqn_agent.DdqnAgent(
                        "time-alpha-ddqn",
                        stock_list,
                        (time_series_feature_set_factory(60), ),
                        action_values=alpha_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-6,
                        use_rolling_bands=False)


        time_series_with_alpha_agent = agents.ddqn_agent.DdqnAgent(
                        "time-rolling-ddqn",
                        stock_list,
                        (time_series_feature_set_factory(60), ),
                        action_values=rolling_band_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-6,
                        use_rolling_bands=True)


        ema_alpha_agent = agents.ddqn_agent.DdqnAgent(
                        "ema-alpha-ddqn",
                        stock_list,
                        (ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)), ),
                        action_values=alpha_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-6,
                        use_rolling_bands=False)

        ema_rolling_agent = agents.ddqn_agent.DdqnAgent(
                        "ema-rolling-ddqn",
                        stock_list,
                        (ema_feature_set_factory((5, 8, 13, 21, 34, 55, 89, 144, 233)), ),
                        action_values=rolling_band_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-6,
                        use_rolling_bands=True)

        fib_agent = agents.ddqn_agent.DdqnAgent(
                        "fib-3-tau-ddqn",
                        stock_list,
                        (bool_fib_feature_set_factory, ),
                        action_values=rolling_band_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-6,
                        cooldown_time=3,
                        use_rolling_bands=False)

        time_series_high_gamma_agent = agents.ddqn_agent.DdqnAgent(
                        "time-series-(1+5.5e-5)-gamma-ddqn",
                        stock_list,
                        (time_series_feature_set_factory(60), ),
                        action_values=rolling_band_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-6,
                        gamma=1 + 5.5e-5,
                        use_rolling_bands=True)

        time_series_low_gamma_agent = agents.ddqn_agent.DdqnAgent(
                        "time-series-(1-4.5e-4)-gamma-ddqn",
                        stock_list,
                        (time_series_feature_set_factory(60), ),
                        action_values=rolling_band_action_values,
                        fc_layer_params=(512, 256, 128),
                        alpha=1e-6,
                        gamma=1 - 4.5e-4,
                        use_rolling_bands=True)


        return [max_log_390,
                max_log_450,
                max_log_585,

                rolling_bands_0_1,
                rolling_bands_0_5,
                rolling_bands_1_0,
                rolling_bands_1_5,
                rolling_bands_2_0,

                catch_rise,

                max_log_deep_agent,
                deep_agent,

                baseline_all_equity,
                baseline_three_quarters_equity,
                baseline_half_equity,
                baseline_tenth_equity,

                alligator,

                time_series_with_alpha_agent,
                time_series_with_rolling_agent,

                mu_ema_rolling_agent,
                time_and_mu_ema_rolling_agent,

                ema_rolling_agent,
                ema_alpha_agent,

                fib_agent,

                time_series_low_gamma_agent,
                time_series_high_gamma_agent]

    def train_agents(self):

        for agent in self.agent_list:
            print("---- Training:", agent.name)
            agent.train(training_iterations=40)
            print("---- Evaluating on Training Set:", agent.name)
            evaluator.EvalEnv(self.stock_list, agent).run_and_save_evaluation("training-set")
            print("---- Evaluating on Eval Set:", agent.name)
            evaluator.EvalEnv(self.eval_stock_list, agent).run_and_save_evaluation()
            print("---- Done with", agent.name)

        print("Done training all agents.")

    def evaluate_agents(self):
        new_agent_list = []
        agent_evaluations = dict()

        for agent in self.agent_list:
            agent.load()
            evaluator.EvalEnv(self.eval_stock_list, agent).run_and_save_evaluation()
            agent_evaluation = agent.get_evaluation()
            agent_evaluations[agent.name] = dict()
            agent_evaluations[agent.name]["evaluation"] = agent_evaluation
            agent_evaluations[agent.name]["is_selected"] = agent_evaluation >= EVALUATION_THRESHOLD

            if agent_evaluations[agent.name]["is_selected"]:
                new_agent_list.append(agent)

        self.agent_list = new_agent_list

    def start_trading(self):
        self.evaluate_agents()
        t = trader.Trader(self.init_equity, self.stock_list, self.agent_list, self.alpaca)
        threading.Thread(target=t.start_live_trading).start()
