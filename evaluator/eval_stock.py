class EvalStock:

    def __init__(self, ticker, prices, env):
        self.ticker = ticker
        self.prices = prices
        self.env = env

    def price(self):
        index = len(self.prices) - self.env.min_length + self.env.index
        return self.prices[index]

    def get_prices(self, length=None):
        index = len(self.prices) - self.env.min_length + self.env.index
        return self.prices[:index]
