from abc import abstractmethod

class AgentBase:

    evaluation = -1.0

    def train(self, training_iterations=None, training_stock_list=None):
        pass

    def load(self):
        pass

    def reset(self, eval_stock_list=None):
        return

    @abstractmethod
    def take_action(self, eval_stock_list=None):
        return {}

    def set_evaluation(self, evaluation):
        self.evaluation = evaluation

    def get_evaluation(self):
        return self.evaluation


