import numpy as np
import matplotlib.pyplot as plt
import util
import sys

from .eval_stock import EvalStock

INIT_EQUITY = 100000
MIN_EVAL_LENGTH = 1500
MIN_INIT_LENGTH = 200

class EvalEnv:

    def __init__(self, stock_list, agent):
        self.agent = agent
        self.stock_list = stock_list

        self.stock_prices = dict()
        for stock in stock_list:
            prices = stock.get_evaluation_dataset()
            if len(prices) > MIN_EVAL_LENGTH + MIN_INIT_LENGTH:
                self.stock_prices[stock.ticker] = prices

        self.reset()

    def step(self):
        action_set = self.agent.take_action(self.eval_stock_list)
        self.index += 1
        done = (self.index == self.min_length - 2)

        future_equity_increase = 1.0
        future_baseline_increase = 1.0

        for stock in self.stock_prices:
            prices = self.stock_prices[stock]
            action = action_set.get(stock, 0)

            index = len(prices) - self.min_length + self.index
            future_price_increase = prices[index + 1] / prices[index]
            future_equity_increase += (future_price_increase - 1) * action
            future_baseline_increase += (future_price_increase - 1) / len(self.stock_prices)

            self.action_sets_over_time.append(action * len(action_set))
            actions_for_stock = self.individual_actions_over_time.get(stock, [])
            actions_for_stock.append(action * len(action_set))
            self.individual_actions_over_time[stock] = actions_for_stock


        self.average_action_over_time.append(sum(action_set.values()))

        self.equity_over_time.append(self.equity_over_time[-1] * future_equity_increase)
        self.baseline_equity_over_time.append(self.baseline_equity_over_time[-1] * future_baseline_increase)
        self.diff_over_time.append(self.equity_over_time[-1] - self.baseline_equity_over_time[-1])

        return done

    def run_and_save_evaluation(self, run_name="final"):
        self.run_name = run_name
        done = False

        # setup toolbar
        print("\n" * 4)
        sys.stdout.write("[%s]" % (" " * 5))
        sys.stdout.flush()
        sys.stdout.write("\b" * (5+1)) # return to start of line, after '['

        while not done:
            done = self.step()
            sys.stdout.write("\b"*5)
            sys.stdout.write(f"{str.zfill(str(self.index), 5)}")
            sys.stdout.flush()

        sys.stdout.write("\n")

        self.save_evaluation()
        self.agent.reset()

    def total_gain(self, values):
        return values[-1] / values[0] - 1

    def sharpe_ratio(self, values, interval=60):
        s, e = 0, interval
        gains = []
        while e < len(values):
            gains.append(values[e] / values[s] - 1)
            s += interval
            e += interval
        return np.mean(gains) / np.std(gains)

    def diff_sharpe_ratio(self, interval=60):
        s, e = 0, interval
        gains = []
        while e < len(self.diff_over_time):
            init_avg = (self.equity_over_time[s] + self.baseline_equity_over_time[s]) / 2
            gains.append((self.diff_over_time[e]) / init_avg - 1)
            s += interval
            e += interval
        return np.mean(gains) / np.std(gains)

    def arithmetic_mean(self, interval=60):
        return np.mean(np.array(self.equity_over_time[interval:]) / np.array(self.equity_over_time[:-interval]) - 1)

    def gain_and_drawdown(self, interval):
        N = len(self.equity_over_time)
        interval_gain = np.power(self.equity_over_time[-1] / self.equity_over_time[0], interval / N)

        lowest_drop = np.float('inf')
        length_of_lowest_drop = N
        for start in range(N - interval):
            for end in range(start + interval, N):
                this_drop = self.equity_over_time[end] / self.equity_over_time[start]
                if this_drop < lowest_drop:
                    lowest_drop = this_drop
                    length_of_lowest_drop = end - start


        lowest_drop = np.power(lowest_drop, interval / length_of_lowest_drop)

        return interval_gain - 1, lowest_drop - 1

    def mean_over_std_log_gains(self):
        gains = np.array(self.equity_over_time[1:]) / np.array(self.equity_over_time[:-1])
        log_gains = np.log(gains)
        return np.mean(log_gains) / np.std(log_gains)

    def save_evaluation(self):
        evaluations = util.load_agent_evaluations()
        evaluations[self.agent.name] = dict()

        total_gain = self.total_gain(self.equity_over_time)
        self.returns = total_gain
        diff_total_gain = self.diff_over_time[-1] / INIT_EQUITY
        hourly_sharpe_ratio = self.sharpe_ratio(self.equity_over_time, 60)
        daily_sharpe_ratio = self.sharpe_ratio(self.equity_over_time, 390)

        mean_hourly_gain = self.arithmetic_mean(60)
        mean_daily_gain = self.arithmetic_mean(390)

        avg_hourly_gain, min_hourly_gain = self.gain_and_drawdown(60)
        avg_daily_gain, min_daily_gain = self.gain_and_drawdown(390)

        evaluation = self.mean_over_std_log_gains()
        self.agent.set_evaluation(evaluation)


        evaluations[self.agent.name]["a) Total Gain"] = total_gain
        evaluations[self.agent.name]["b) Hourly Sharpe Ratio"] = hourly_sharpe_ratio
        evaluations[self.agent.name]["c) Daily Sharpe Ratio"] = daily_sharpe_ratio\

        evaluations[self.agent.name]["d) Avg Daily Gain"] = avg_daily_gain
        evaluations[self.agent.name]["e) Min Daily Gain"] = min_daily_gain
        evaluations[self.agent.name]["f) Daily Avg / Min"] = avg_daily_gain / min_daily_gain
        evaluations[self.agent.name]["g) Daily exp(Avg / Min)"] = np.exp(avg_daily_gain / min_daily_gain)

        evaluations[self.agent.name]["h) E[log(G)]/sqrt(V[log(G)])"] = evaluation
        evaluations = util.save_agent_evaluations(evaluations)


        print("\n---- Evaluation for", self.agent.name)
        print("\t\tTotal Gain               : {:.4f}%".format(100 * total_gain))
        print("\t\tHourly Sharpe Ratio      : {:.4f}".format(hourly_sharpe_ratio))
        print("\t\tDaily Sharpe Ratio       : {:.4f}".format(self.sharpe_ratio(self.equity_over_time, 390)))

        print("\t\tAvg Hourly Gain          : {:.4f}%".format(100 * avg_hourly_gain))
        print("\t\tMin Hourly Gain          : {:.4f}%".format(100 * min_hourly_gain))
        print("\t\tHourly Avg / Min         : {:.4f}".format(avg_hourly_gain / min_hourly_gain))
        print("\t\tEvaluation.              : {:.4f}".format(evaluation))

        fig = plt.figure()

        ax = fig.add_subplot(511)
        ax.set_title(("Equity -- Gain: {:.4f}%, "+\
                      "ADG: {:.4f}%, MDG: {:.4f}%, WDG: {:.4f}%,\n"+\
                      "-ADG/WDG: {:.4f}, -MDG/WDG: {:.4f}, eval: {:.4f}").format(
                                    100 * total_gain,
                                    100 * avg_daily_gain,
                                    100 * mean_daily_gain,
                                    100 * min_daily_gain,
                                    -avg_daily_gain / min_daily_gain,
                                    -mean_daily_gain / min_daily_gain,
                                    evaluation))

        ax.plot(100 * np.array(self.equity_over_time) / INIT_EQUITY)
        ax.plot(100 * np.array(self.baseline_equity_over_time) / INIT_EQUITY, alpha=0.5)

        bx = fig.add_subplot(512)
        bx.set_title("Equity Difference -- Gain: {:.4f}".format(100 * self.diff_over_time[-1] / INIT_EQUITY))
        bx.plot(100 * np.array(self.diff_over_time) / INIT_EQUITY)

        cx = fig.add_subplot(513)
        cx.set_title("Actions Distribution")
        cx.hist(self.action_sets_over_time, bins=[0.01 * i for i in range(102)])

        dx = fig.add_subplot(514)
        dx.set_title("Actions Over Time")
        dx.plot(self.average_action_over_time)

        ex = fig.add_subplot(515)
        ex.set_title("Individual Actions Over Time of Best and Worst")
        best_stock, highest_action = None, 0.0
        worst_stock, lowest_action = None, 1.0
        for stock in self.stock_prices:
            avg_action = float(np.mean(self.individual_actions_over_time[stock]))

            if highest_action <= avg_action:
                best_stock = stock
                highest_action = avg_action

            if lowest_action >= avg_action:
                worst_stock = stock
                lowest_action = avg_action
        ex.plot(self.individual_actions_over_time[best_stock])
        ex.plot(self.individual_actions_over_time[worst_stock])

        plt.tight_layout()
        plt.savefig("training_data_progress/" + self.agent.name + "-" + self.run_name + '.png')
        plt.close('all')

    def reset(self):
        self.eval_stock_list = [EvalStock(ticker, self.stock_prices[ticker], self) for ticker in self.stock_prices]

        self.index = MIN_INIT_LENGTH
        self.min_length = min([len(v) for v in self.stock_prices.values()])
        self.equity_over_time = [INIT_EQUITY]
        self.baseline_equity_over_time = [INIT_EQUITY]
        self.diff_over_time = [0]

        self.agent.reset(self.eval_stock_list)
        self.action_sets_over_time = []
        self.average_action_over_time = []
        self.individual_actions_over_time = dict()
