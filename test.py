import runner

def main():
    """ Runs the main train-evaluate-trade loop.
    """
    r = runner.Runner()
    r.train_agents()


if __name__ == '__main__':
    main()