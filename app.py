import json
from flask import Flask, request, jsonify, send_file
import util
import train
import os.path
from os import path

app = Flask(__name__)

# A welcome message to test our server
@app.route('/', methods=['GET'])
def index():
    result = []
    profits = util.load_current_profit_list()
    for stock in profits:
        result.append(stock + ": " + "{:.5f} %".format(100 * profits[stock]))
    positions = "<div>" + "</div><div>".join(result) + "</div>"

    return "<h1>RL Trader -- Current Profits</h1>" + positions

# Lists out training evaluation
@app.route('/evaluations', methods=['GET'])
def evaluations():
    return "<h2>Agent Evaluations</h2><pre>" + json.dumps(util.load_agent_evaluations(), sort_keys=True, indent=4) + "</pre>"

# Lists out agent info
@app.route('/info', methods=['GET'])
def info():
    return "<h2>Agent Info</h2><pre>" + json.dumps(util.load_live_trading_stocks_info(), sort_keys=True, indent=4) + "</pre>"

@app.route('/curr_training')
def get_image():
    if request.args.get('step') == 'FIRST':
       filename = '/root/stock-trading/training_data_progress/trainstep_FIRST.png'
    else:
       filename = '/root/stock-trading/training_data_progress/trainstep_LAST.png'
    return send_file(filename, mimetype='image/png')

@app.route('/start_training', methods=['GET'])
def start_training():
    train.create_datasets()
    train.start_training()




if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, host='0.0.0.0', port=5000)
