from ..agent_base import AgentBase


class Baseline(AgentBase):

    def __init__(self, name, stock_list, alpha=1.0):
        self.name = name
        self.stock_list = stock_list
        self.alpha = alpha

    def take_action(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        action_set = dict()
        for stock in stock_list:
            action_set[stock.ticker] = self.alpha / len(stock_list)

        return action_set
