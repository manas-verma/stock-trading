from .trainer import Trainer

import numpy as np
import json
import os
import util
import yfinance as yf
from hurst import random_walk

TRAINING_ITERATIONS = 15

def train(stocks, training_iterations=TRAINING_ITERATIONS):
    try:
        os.mkdir('training_data_progress')
        os.mkdir('training_data_progress/train')
        os.mkdir('training_data_progress/eval')
    except FileExistsError:
        pass
    dataset = load_dataset(stocks)
    trainer = Trainer(dataset)
    trainer.train_all(training_iterations)


def load_dataset(stocks):
    dataset = {}
    for stock in stocks:
        with open("training/datasets/" + stock + ".json", "r") as file:
            dataset[stock] = json.load(file)
            dataset[stock] = list(map(np.float, dataset[stock]))
    return dataset


def generate_dataset(stocks):
    for stock in stocks:
        with open("training/datasets/" + stock + ".json", "w+") as file:
            file.write(json.dumps(get_data_for_stock(stock)))


def get_data_for_stock(stock):
    if stock == "SINE_WAVE":
        return [str(50 + 30 * np.sin(i/10)) for i in range(1200)]
    elif stock == "NOISY_SINE_WAVE":
        return [str(45 + 30 * np.sin(i/10) + 10 * np.random.random()) for i in range(1200)]
    elif stock == "LINEAR_RISING_SINE_WAVE":
        return [str(50 + 30 * np.sin(i/10) + i / 10) for i in range(1200)]
    elif stock == "LINEAR_RISING_NOISY_SINE_WAVE":
        return [str(50 + 30 * np.sin(i/10) + i / 10 + 10 * np.random.random()) for i in range(1200)]
    elif stock == "LINEAR_FALLING_NOISY_SINE_WAVE":
        return [str(150 + 30 * np.sin(i/10) - i / 10 + 10 * np.random.random()) for i in range(1200)]
    elif stock == "EXP_RISING_SINE_WAVE":
        return [str(10 * np.random.random() + 50 + 30 * np.sin(i/10) + 150 * np.exp(i - 1200)) for i in range(1200)]
    elif stock == "DOWN_UP_SINE_WAVE":
        return [str(10 * np.random.random() + 150 + 30 * np.sin(i/10) + (-i / 10 if i < 600 else -60 + i/10) ) for i in range(1200)]
    elif stock == "RANDOM_WALK_PERSISTENT":
        return [str(val) for val in random_walk(1200, proba=0.6)]
    elif stock == "RANDOM_WALK_RANDOM":
        return [str(val) for val in random_walk(1200, proba=0.5)]
    elif stock == "RANDOM_WALK_ANTIPERSISTENT":
        return [str(val) for val in random_walk(1200, proba=0.4)]
    elif stock == "RANDOM_WALK_MIX":
        N = 1200
        M = 100
        return list(np.array([[str(max(0.01, val * 5+ 150)) for val in random_walk(M, proba=min(0.99, max(0.01, (np.random.normal() + 0.5) / 4)))] for _ in range(N // M)]).flatten())
    else:
        data = yf.download(tickers=stock, period="7d", interval="1m")
        return [str(price) for price in data.Close]

