import live_trading
import training
import util
from app import app
from flask_ngrok import run_with_ngrok
import sys
import train

def start_live_trading(self=None):
    print(self)
    live_trading.start_live_trading(util.load_live_trading_stocks_info(), post_trading_callback, pre_training_callback)


def close_all_positions(self=None):
    print(self)
    live_trading.close_all_positions()

def pre_training_callback(self=None):
    train.create_datasets()
    train.start_training()

def post_trading_callback(self=None):
    close_all_positions()
    train.create_datasets()
    train.start_training()



if __name__ == '__main__':
    # start_live_trading("This is a message.")
    close_all_positions("Closing All Positions")
    start_live_trading("Starting Live Trading")

    if sys.argv[0].lower() in ('colab', 'google-colab', 'google', 'notebook'):
        run_with_ngrok(app)
        app.run()
    else:
        app.run(threaded=True, host='0.0.0.0', port=5000)
