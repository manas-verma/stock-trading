from .trader import Trader
import util

from threading import Thread


def start_live_trading(stocks, callback_after_close, pre_training_callback):
    trader = Trader(stocks, callback_after_close, pre_training_callback)
    Thread(target=trader.start_live_trading).start()


def close_all_positions():
    alpaca = util.get_alpaca()
    alpaca.cancel_all_orders()
    positions = alpaca.list_positions()
    for position in positions:
        if position.side == "long":
            side = "sell"
        else:
            side = "buy"
        qty = abs(int(float(position.qty)))
        try:
            alpaca.submit_order(position.symbol, qty, side, "market", "day")
        except Exception as e:
            print("Caught Exception:", e)
            print("Order did not go through in attempt to close all positions.")
