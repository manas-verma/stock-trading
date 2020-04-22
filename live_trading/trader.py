from stock import Stock
import util

import threading
import time
import datetime


TRADE_DURING_OPEN_HOURS_ONLY = True
TOTAL_EQUITY = 50000

class Trader:

    def __init__(self, tickers=(), callback_after_close=None, pre_training_callback=None):
        print("Initializing Trader...")
        self.alpaca = util.get_alpaca()

        print("With stocks:", list(tickers.keys()))
        self.stocks = [Stock(ticker,
                        self,
                        TOTAL_EQUITY * tickers[ticker]["equity_weight"],
                        tickers[ticker]["weight_master_action"])
                    for ticker in tickers]

        self.callback_after_close = callback_after_close
        self.pre_training_callback = pre_training_callback

    def is_market_open(self):
        return self.alpaca.get_clock().is_open

    def will_market_close_soon(self):
        # Figure out when the market will close so we can prepare to sell
        # beforehand.
        clock = self.alpaca.get_clock()
        closing_time = clock.next_close.replace(tzinfo=datetime.timezone.utc)
        curr_time = clock.timestamp.replace(tzinfo=datetime.timezone.utc)
        return (closing_time.timestamp() - curr_time.timestamp()) < (60 * 15)

    def will_market_open_soon(self):
        # Figure out when the market will close so we can prepare to sell
        # beforehand.
        clock = self.alpaca.get_clock()
        opening_time = clock.next_open.replace(tzinfo=datetime.timezone.utc)
        curr_time = clock.timestamp.replace(tzinfo=datetime.timezone.utc)
        return (opening_time.timestamp() - curr_time.timestamp()) < (60 * 60 * 2)

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
            is_open = self.alpaca.get_clock().is_open

    def start_live_trading(self):
        if not self.load_stocks():
            self.pre_training_callback()

        while True:
            util.save_current_profit_list(self.stocks)

            # Wait for market to open.
            if TRADE_DURING_OPEN_HOURS_ONLY and not self.is_market_open():
                if self.callback_after_close is not None and not self.will_market_open_soon():
                    self.callback_after_close()
                print("Waiting for market to open...")
                self.wait_till_market_opens()
                print("Market has opened!")

            # If market is closing soon, then close all positions.
            if TRADE_DURING_OPEN_HOURS_ONLY and self.will_market_close_soon():
                print("Sleeping until market closes (15 minutes).")
                if self.callback_after_close is not None:
                    self.callback_after_close()
                    self.callback_after_close = None
                time.sleep(60 * 15)
                continue

            # Start a tick, which will kick off multiple threads. Wait for all
            # threads to complete.
            self.tick()
            time.sleep(59)

    def load_stocks(self):
        all_loaded = True
        for stock in self.stocks:
            all_loaded = all_loaded and stock.load_policy()
        return all_loaded

    def equity(self):
        return self.alpaca.get_account().equity

    def tick(self):
        for stock in self.stocks:
            threading.Thread(target=stock.tick).start()
