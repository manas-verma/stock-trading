import app
import runner
import flask_ngrok

import schedule
import time
import sys
import threading

def schedule_every_weekday_at(utc_time, job):
    schedule.every().monday.at(utc_time).do(job)
    schedule.every().tuesday.at(utc_time).do(job)
    schedule.every().wednesday.at(utc_time).do(job)
    schedule.every().thursday.at(utc_time).do(job)
    schedule.every().friday.at(utc_time).do(job)

def schedule_every_weeknight_at(utc_time, job):
    schedule.every().sunday.at(utc_time).do(job)
    schedule.every().monday.at(utc_time).do(job)
    schedule.every().tuesday.at(utc_time).do(job)
    schedule.every().wednesday.at(utc_time).do(job)
    schedule.every().thursday.at(utc_time).do(job)

def get_arg():
    if len(sys.argv) > 1:
        return sys.argv[1].lower()
    return ""


def run_app():
    if get_arg() in ('colab', 'google-colab', 'google', 'notebook'):
        flask_ngrok.run_with_ngrok(app)
        app.app.run()
    else:
        app.app.run(threaded=True, host='0.0.0.0', port=5000)


def main():
    """ Runs the main train-evaluate-trade loop.
    """
    r = runner.Runner()
    if get_arg() in ('--train-now'):
        r.train_agents()

    if get_arg() in ('--trade-now'):
        r.start_trading()

    schedule_every_weeknight_at("20:45", r.train_agents)
    schedule_every_weekday_at("11:00", r.start_trading)

    while True:
       schedule.run_pending()
       time.sleep(300)


if __name__ == '__main__':
    threading.Thread(target=main).start()
    threading.Thread(target=run_app).start()
