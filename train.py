import live_trading
import training
import selector
import util
import sys

def create_datasets(self=None):
    print(self)
    training.generate_dataset(_get_stock_list())


def start_training(self=None):
    print(self)
    training.train(_get_stock_list())
    selector.Selector().select_stocks()


def _get_stock_list():
    if __name__ == '__main__' and len(sys.argv) > 1:
        stock_list = sys.argv[1:]
    else:
        stock_list = util.get_training_set_tickers()
        stock_list.extend(util.get_stock_universe_list())
    return stock_list


if __name__ == '__main__':
    # start_live_trading("This is a message.")
    create_datasets("Creating Dataset")
    start_training("Starting Training")
