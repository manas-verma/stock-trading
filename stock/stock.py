from agent import Agent
from feature_generator import FeatureGenerator
import util

from tf_agents.trajectories.time_step import TimeStep

import numpy as np
import tensorflow as tf

DISCRETE = False

class Stock:

    def __init__(self, ticker, trader, equity, weight_master_action=0.0):
        self.ticker = ticker
        self.trader = trader
        self.discrete = DISCRETE
        self.alpaca = trader.alpaca

        self.init_equity = equity
        self.num_shares = self.quantity()
        self.buying_power = equity - self.num_shares * self.current_price()

        self.weight_master_action = weight_master_action
        self.feature_generator = FeatureGenerator(self.prices(300))

    def load_policy(self):
        try:
            self.master_agent_policy = tf.saved_model.load("model_data/MASTER")
            self.special_agent_policy = tf.saved_model.load("model_data/" + self.ticker)
            return True
        except Exception as e:
            print(e)
            return False

    def equity(self):
        return self.buying_power + self.current_price() * self.num_shares

    def tick(self):
        # Update feature generator with price.
        price = self.current_price()
        self.feature_generator.append_price(price)

        # Take action based on features.
        dynamic_features = self.get_dynamic_features()
        features = self.feature_generator.get_features(dynamic_features)
        time_step = TimeStep(
                step_type=np.array([1], dtype=np.int32),
                reward=np.array([1.0], dtype=np.float32),
                discount=np.array([1.001], dtype=np.float32),
                observation=np.array([features], dtype=np.float32))
        master_action = float(self.master_agent_policy.action(time_step)[0])
        special_action = float(self.special_agent_policy.action(time_step)[0])

        alpha = self.weight_master_action
        action = alpha * master_action + (1 - alpha) * special_action

        self.buy_or_sell_shares(action, price)

    def buy_or_sell_shares(self, action, price):
        # |action| represents the percentage of equity that should be put into shares.
        if self.discrete:
            action = [0, 0.25, 0.5, 0.75, 1.0][int(action)]
        else:
            action = 0.5 + action / 2
        num_shares_to_buy = (self.equity() * action) / price - self.num_shares

        self.buying_power -= num_shares_to_buy * price
        self.num_shares += num_shares_to_buy

        # If fractional shares are not allowed, then adjust the number of shares.
        fractional_shares = self.num_shares % 1
        self.num_shares = self.num_shares // 1
        self.buying_power += fractional_shares * price
        num_shares_to_buy = num_shares_to_buy // 1
        if np.random.random() < fractional_shares and self.buying_power > price:
            self.num_shares += 1
            num_shares_to_buy += 1
            self.buying_power -= price

        if num_shares_to_buy > 0:
            self.buy(num_shares_to_buy)
        if num_shares_to_buy < 0:
            self.sell(abs(num_shares_to_buy))

    def get_dynamic_features(self):
        return np.array([self.current_price() * self.quantity() / self.equity()])

    def __eq__(self, other):
        return self.ticker == other.ticker and self.num_shares == other.num_shares

    def copy(self):
        return Stock(self.ticker)

    def quantity(self):
        try:
            qty = float(self.alpaca.get_position(self.ticker).qty)
            if self.num_shares != qty:
                equity = self.equity()
                print("Discrepency between actual quantity and expected quantity.")
                print("\tActual: ", qty)
                print("\tExpected: ", self.num_shares)
                self.num_shares = qty
                self.buying_power = equity - self.current_price() * qty
            return qty
        except Exception as e:
            print("Caught Exception:", e)
            return 0

    def current_price(self):
        try:
            return float(self.alpaca.get_position(self.ticker).current_price)
        except Exception as e:
            print("Caught Exception:", e)
            barset = self.alpaca.get_barset(self.ticker, 'minute', limit=1)
            return barset[self.ticker][0].c

    def profit(self):
        try:
            return float(self.alpaca.get_position(self.ticker).unrealized_pl)
        except Exception as e:
            print("Caught Exception:", e)
            return 0

    def relative_profit(self):
        try:
            return float(self.alpaca.get_position(self.ticker).unrealized_plpc)
        except Exception as e:
            print("Caught Exception:", e)
            return 0

    def prices(self, length=200):
        barset = self.alpaca.get_barset(self.ticker, 'minute', limit=length)
        return np.array([bar.c for bar in barset[self.ticker]])

    def volume(self, length=200):
        barset = self.alpaca.get_barset(self.ticker, 'minute', limit=length)
        return np.array([bar.v for bar in barset[self.ticker]])

    def buy(self, qty=1):
        status = {'completed': False}
        self.submit_order(abs(qty), True, status)
        return status

    def sell(self, qty=None):
        status = {'completed': False}
        qty = qty if qty else self.quantity()
        self.submit_order(abs(qty), False, status)
        return status

    def submit_order(self, qty, is_buy_order, status):
        side = "buy" if is_buy_order else "sell"
        if qty > 0:
            try:
                self.alpaca.submit_order(self.ticker, qty, side,
                                         "market", "day")
                print("Market order of | " + str(qty) + " " + self.ticker +
                      " " + side + " | completed.")
                status['completed'] = True
            except Exception:
                print("Order of | " + str(qty) + " " + self.ticker + " " +
                      side + " | did not go through.")
                status['completed'] = False
        else:
            print("Quantity is 0, order of | " + str(qty) + " " + self.ticker +
                  " " + side + " | not completed.")
            status['completed'] = True
