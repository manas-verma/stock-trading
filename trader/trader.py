import util

import threading
import time
import datetime
import numpy as np


class Trader:

    def __init__(self, init_equity, stock_list, agent_list, alpaca):
        self.alpaca = alpaca
        self.buying_power = self.init_equity = init_equity
        self.stock_list = stock_list
        self.agent_list = agent_list

        self.agent_action_weights = self.create_agent_action_weights()

        # Calculate our buying power given the positions for each stock.
        for stock in self.stock_list:
            if stock.quantity == 0:
                self.buying_power -= stock.quantity() * stock.price()

    def equity(self):
        return self.buying_power + sum([stock.quantity() + stock.price() for stock in self.stock_list])

    def create_agent_action_weights(self):
        """ Simple weighted average of agent evaluations.
        """
        agent_action_weights = dict()

        total_weight = 0
        for agent in self.agent_list:
            agent_evaluation = agent.get_evaluation()
            agent_action_weights[agent.name] = agent_evaluation
            total_weight += agent_evaluation

        for agent in self.agent_list:
            agent_action_weights[agent.name] /= total_weight

        return agent_action_weights

    def is_market_open(self):
        return self.alpaca.get_clock().is_open

    def will_market_close_soon(self):
        # Figure out when the market will close so we can prepare to sell
        # beforehand.
        clock = self.alpaca.get_clock()
        closing_time = clock.next_close.replace(tzinfo=datetime.timezone.utc)
        curr_time = clock.timestamp.replace(tzinfo=datetime.timezone.utc)
        return (closing_time.timestamp() - curr_time.timestamp()) < (60 * 15)

    def wait_till_market_opens(self):
        print("Market Closed, Starting Training")
        is_open = self.is_market_open()
        while not is_open:
            clock = self.alpaca.get_clock()
            opening_time = \
                clock.next_open.replace(tzinfo=datetime.timezone.utc)
            curr_time = clock.timestamp.replace(tzinfo=datetime.timezone.utc)
            time_diff = (opening_time.timestamp() - curr_time.timestamp())
            time_to_open = int(time_diff / 60)
            print(str(time_to_open) + " minutes till market opens.")
            time.sleep(60)
            is_open = self.is_market_open()

    def start_live_trading(self):
        # Load all agents before trading.
        print("Loading agents.")
        self.load_agents()

        print("Starting to trade.")
        while True:
            # Wait for market to open.
            if not self.is_market_open():
                print("Waiting for market to open...")
                self.wait_till_market_opens()
                print("Market has opened!")

            # If market is closing soon, then close all positions and end trading session.
            if self.will_market_close_soon():
                self.close_all_positions()
                print("Sleeping until market closes (15 minutes).")
                time.sleep(60 * 15)
                return

            # Start a tick, which will kick off multiple threads. Wait for all
            # threads to complete.
            self.tick()
            time.sleep(59)

    def close_all_positions(self):
        self.alpaca.cancel_all_orders()
        for stock in self.stock_list:
            stock.close_position()

    def load_agents(self):
        for agent in self.agent_list:
            agent.load()

    def tick(self):
        print("Tick...")
        equity = self.equity()

        # The action represents the percentage of equity that should be put into shares.
        action_per_stock = dict()

        for agent in self.agent_list:
            action_set = agent.take_action()

            # Update the action for each stock according to the agent's weight.
            for ticker in action_set:
                action = action_per_stock.get(ticker, 0.0)
                action += action_set[ticker] * self.agent_action_weights[agent.name]
                action_per_stock[ticker] = action

        print(action_per_stock)

        threads = []
        for stock in self.stock_list:
            action = action_per_stock.get(stock.ticker, 0.0)
            if  action <= 0.0:
                continue

            quantity = stock.quantity()
            price = stock.price()


            num_shares_to_buy = (equity * action) / price - quantity

            self.buying_power -= num_shares_to_buy * price
            quantity += num_shares_to_buy

            # If fractional shares are not allowed, then adjust the number of shares.
            fractional_shares = quantity % 1
            self.buying_power += fractional_shares * price
            num_shares_to_buy = num_shares_to_buy // 1

            # If we can afford it, buy an extra share with probability of the fraction.
            if self.buying_power > price and np.random.random() < fractional_shares:
                num_shares_to_buy += 1
                self.buying_power -= price

            print(stock.ticker, ":", num_shares_to_buy)
            target = lambda: stock.buy_shares(num_shares_to_buy)
            thread = threading.Thread(target=target)
            thread.start()
            threads.append(thread)

        for thread in threads:
            thread.join()
