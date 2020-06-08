from ..agent_base import AgentBase


class CatchRise(AgentBase):

    def __init__(self, name, stock_list):
        self.name = name
        self.stock_list = stock_list
        self.reset(self.stock_list)

    def reset(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        self.threshold_price = dict()
        self.previous_prices = dict()
        for stock in stock_list:
            price = stock.get_prices(2)[-1]
            self.threshold_price[stock.ticker] = price
            self.previous_prices[stock.ticker] = price

    def load(self):
        self.reset()

    def take_action(self, eval_stock_list=None):
        stock_list = self.stock_list
        if eval_stock_list is not None:
            stock_list = eval_stock_list

        action_set = dict()
        for stock in stock_list:
            action = self.take_action_for_stock(stock)
            action_set[stock.ticker] = action / len(stock_list)

        return action_set

    def take_action_for_stock(self, stock):
        price = stock.price()

        previous = self.previous_prices[stock.ticker]
        self.previous_prices[stock.ticker] = price

        threshold = self.threshold_price[stock.ticker]
        self.threshold_price[stock.ticker] = min(threshold, 0.995 * price)

        if price > threshold and price > 0.995 * previous:
            return 0.5

        if price > threshold and price < 0.995 * previous:
            return 0.1

        if price <= threshold:
            return 0.0

        return 0.1



