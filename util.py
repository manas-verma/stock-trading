import alpaca_trade_api
import json

ALPACA_API_KEY = "PK33XATVTEZDKRBFE08Y"
ALPACA_API_SECRET = "8HJvuAF5nSfJuicIDfB78TwSzURo1OKzs4NUoR4G"
APCA_API_BASE_URL = "https://paper-api.alpaca.markets"


def get_alpaca():
    return alpaca_trade_api.REST(ALPACA_API_KEY,
                         ALPACA_API_SECRET,
                         APCA_API_BASE_URL,
                         'v2')

def get_training_set_tickers():
    return [
        "NOISY_SINE_WAVE",
        # "SINE_WAVE",
        # "LINEAR_RISING_SINE_WAVE",
        "LINEAR_FALLING_NOISY_SINE_WAVE",
        # "LINEAR_RISING_NOISY_SINE_WAVE",
        # "EXP_RISING_SINE_WAVE",
        "DOWN_UP_SINE_WAVE"
    ]

def get_stock_universe_list():
    stocks = {"stocks": []}
    with open("stock_universe.json", "r") as file:
        stocks = json.load(file)
    return stocks["stocks"]

def get_eval_stock_list():
    stocks = {"eval_stocks": []}
    with open("stock_universe.json", "r") as file:
        stocks = json.load(file)
    return stocks["eval_stocks"]

def save_live_trading_stocks_info(stocks_info):
    with open("live_trading_stocks_info.json", "w+") as file:
        file.write(json.dumps(stocks_info, sort_keys=True, indent=4))

def load_live_trading_stocks_info():
    stocks = dict()
    with open("live_trading_stocks_info.json", "r") as file:
        stocks = json.load(file)
    return stocks

def save_current_profit_list(stocks):
    result = dict()
    total_init_equity = 0
    total_curr_equity = 0
    for stock in stocks:
        result[stock.ticker] = (stock.equity() / stock.init_equity) - 1
        total_init_equity += stock.init_equity
        total_curr_equity += stock.equity()
    result["TOTAL EQUITY"] = (total_curr_equity / total_init_equity) - 1
    with open("current_stock_profit_list.json", "w+") as file:
        file.write(json.dumps(result, sort_keys=True, indent=4))

def load_current_profit_list():
    stocks = dict()
    with open("current_stock_profit_list.json", "r") as file:
        stocks = json.load(file)
    return stocks

def save_agent_evaluations(evaluations):
    with open("agent_evaluations.json", "w+") as file:
        file.write(json.dumps(evaluations, sort_keys=True, indent=4))

def load_agent_evaluations():
    evaluations = dict()
    with open("agent_evaluations.json", "r") as file:
        evaluations = json.load(file)
    return evaluations

def save_training_report(report):
    with open("training_report.json", "w+") as file:
        file.write(json.dumps(report, sort_keys=True, indent=4))

def load_training_report():
    report = dict()
    with open("training_report.json", "r") as file:
        report = json.load(file)
    return report
