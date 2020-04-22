import numpy as np
import matplotlib.pyplot as plt
from hurst import compute_Hc

MIN_REWARD = -100
MAX_REWARD = 100

INIT_EQUITY = 10000

DISCRETE = False

MOVING_EQUITY_AVERAGE_WINDOW = 20

RUNNING_LENGTH_TO_CONSIDER_FOR_EVALUATION = int(6.5 + 60)  # Number of minutes that the stock market is open.

LOG_DIFF_REWARD_WEIGHT = 1
FRAC_DIFF_REWARD_WEIGHT = 0
DEVIATION_REWARD_WEIGHT = 0


class RewardGenerator:

    def __init__(self, prices, index):
        self.prices = prices
        self.index = index
        self.init_index = index
        self.current_reward = 0

        self.buying_power_over_time = [INIT_EQUITY, INIT_EQUITY]
        self.equity_over_time = [INIT_EQUITY, INIT_EQUITY]
        self.num_shares_over_time = [0, 0]
        self.action_over_time = [0, 0]

        self.equity_mean = INIT_EQUITY
        self.equity_var = 0.05 * INIT_EQUITY
        self.equity_alpha = 2 / (MOVING_EQUITY_AVERAGE_WINDOW + 1)

        self.init_equity = INIT_EQUITY
        self.buying_power = INIT_EQUITY
        self.num_shares = 0

    def get_reward(self, action):
        self.buy_or_sell_shares(action)

        r1 = self.get_future_log_diff() * LOG_DIFF_REWARD_WEIGHT
        r2 = self.get_future_percent_diff() * FRAC_DIFF_REWARD_WEIGHT
        r3 = self.get_future_equity_deviation() * DEVIATION_REWARD_WEIGHT

        self.current_reward = r1 + r2 + r3

        self.buying_power_over_time.append(self.buying_power)
        self.equity_over_time.append(self.equity())
        self.num_shares_over_time.append(self.num_shares)
        self.action_over_time.append(action)
        self.index += 1
        return self.current_reward

    def buy_or_sell_shares(self, action):
        current_price = self.prices[self.index]
        # |action| represents the percentage of equity that should be put into shares.
        if DISCRETE:
            action = [0, 0.25, 0.5, 0.75, 1.0][int(action)]
        else:
            action = 0.5 + action / 2

        num_shares_to_buy = (self.equity() * action) / current_price - self.num_shares

        self.buying_power -= num_shares_to_buy * current_price
        self.num_shares += num_shares_to_buy

        # If fractional shares are not allowed, then adjust the number of shares.
        fractional_shares = self.num_shares % 1
        self.num_shares = self.num_shares // 1
        self.buying_power += fractional_shares * current_price
        if np.random.random() < fractional_shares and self.buying_power > current_price:
            self.num_shares += 1
            self.buying_power -= current_price

    def get_future_equity(self):
        future_price = self.prices[self.index + 1] if self.index + 1 < len(self.prices) else self.prices[-1]
        return self.buying_power + future_price * self.num_shares

    def get_future_log_diff(self):
        return np.log(self.get_future_equity() / self.equity())

    def get_future_percent_diff(self):
        return (self.get_future_equity() - self.equity()) / self.equity()

    def get_future_equity_deviation(self):
        diff = self.equity() - self.equity_mean
        self.equity_mean += self.equity_alpha * diff
        self.equity_var += self.equity_alpha * diff * diff
        self.equity_var *= (1 - self.equity_alpha)

        return (self.get_future_equity() - self.equity_mean) / np.sqrt(self.equity_var)



    def value_in_shares(self):
        i = -1 if self.index == len(self.prices) else self.index
        return self.num_shares * self.prices[i]

    def equity(self):
        i = -1 if self.index == len(self.prices) else self.index
        return self.buying_power + self.num_shares * self.prices[i]

    def get_dynamic_features(self):
        return [self.value_in_shares() / self.equity()]

    def render(self):
        plt.figure()
        plt.plot(self.equity_over_time)
        plt.title("Equity")
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        im = Image.open(buf)
        result = np.array(im).copy()
        plt.close('all')
        buf.close()
        return result

    def save_run_fig(self, filename):
        i = len(self.prices)-1 if self.index == len(self.prices) else self.index
        price_list = self.prices[self.init_index-2:i+1]

        action_over_time = [self.action_over_time[3]] * 3 + self.action_over_time[3:]

        indices = [311,312,313]
        names = ["Equtiy", "Prices", "Action"]
        data_list = [self.equity_over_time, price_list, action_over_time]

        fig = plt.figure()
        for index, name, data in zip(indices, names, data_list):
            ax = fig.add_subplot(index)
            ax.set_title(name)
            ax.plot(data)
        plt.savefig(filename + '.png')
        plt.close('all')

    def get_evaluation(self):
        evaluation = dict()
        profits = np.array(self.equity_over_time) / self.init_equity - 1

        evaluation["final_profit"] = self.equity() / self.init_equity - 1
        evaluation["min_profit"] = min(profits)
        evaluation["max_profit"] = max(profits)
        evaluation["duration_in_negative"] = len([e < self.init_equity for e in self.equity_over_time])

        longest_fall, longest_rise = 0, 0
        current_fall, current_rise = 0, 0
        for e_1, e_2 in zip(self.equity_over_time[:-1], self.equity_over_time[1:]):
            if e_1 > e_2:
                current_rise = 0
                current_fall += 1
                longest_fall = max(current_fall, longest_fall)
            else:
                current_fall = 0
                current_rise += 1
                longest_rise = max(current_rise, longest_rise)
        evaluation["longest_fall"] = longest_fall
        evaluation["longest_rise"] = longest_rise

        evaluation["profit_avg"] = np.mean(profits)
        evaluation["profit_std"] = np.std(profits)

        H, c, _ = compute_Hc(profits, kind='change', simplified=False)

        evaluation["hurst_exponent"] = H
        evaluation["hurst_constant"] = c


        evaluation["avg_to_rs_ratio"] = evaluation["profit_avg"] / (c * np.power(RUNNING_LENGTH_TO_CONSIDER_FOR_EVALUATION, H))
        return evaluation