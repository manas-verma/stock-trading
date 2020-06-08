import util
import numpy as np

EQUITY_WEIGHT_THRESHOLD = -0.04

class Selector:

    def __init__(self):
        self.evaluations = util.load_agent_evaluations()

        # Ignore names in the form "Master:TSLA"
        self.tickers = [ticker for ticker in self.evaluations \
                if ":" not in ticker and \
                ticker not in util.get_training_set_tickers()]

        self.stocks_info = dict()

    def get_special_agent_evaluation(self, ticker):
        return self.evaluations[ticker]

    def get_master_agent_evaluation(self, ticker):
        return self.evaluations["Master:" + ticker]

    def get_ranking(self, evaluation):
        return evaluation["avg_to_rs_ratio"]

    def select_stocks(self):

        total_weight = 0
        for ticker in self.tickers:
            special_ranking = self.get_ranking(self.get_special_agent_evaluation(ticker))
            master_ranking = self.get_ranking(self.get_master_agent_evaluation(ticker))
            weight_master_action = np.exp(master_ranking) / (np.exp(master_ranking) + np.exp(special_ranking))

            equity_weight = (master_ranking * np.exp(master_ranking) + special_ranking * np.exp(special_ranking)) \
                                    / (np.exp(master_ranking) + np.exp(special_ranking))
            if equity_weight < EQUITY_WEIGHT_THRESHOLD:
                continue

            equity_weight += EQUITY_WEIGHT_THRESHOLD + 0.01

            self.stocks_info[ticker] = {
                "equity_weight": equity_weight,
                "weight_master_action": weight_master_action
            }

            total_weight += equity_weight

        for ticker in self.tickers:
            self.stocks_info[ticker]["equity_weight"] /= total_weight


        util.save_live_trading_stocks_info(self.stocks_info)

