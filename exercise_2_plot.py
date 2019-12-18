import os
import numpy as np
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt


EXPERIMENTS_PATH = "a2"
EXPERIMENTS = ["experiment_" + str(i+1) for i in range(20)]
INFO_FILE = "info.txt"
LOSSES_FILE = "losses.txt"


def plot_losses(losses, title, path):
    plt.figure()
    plt.plot(losses)
    plt.title(title)
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.savefig(os.path.join(path, "loss.jpg"))


def main():
    for experiment in EXPERIMENTS:
        path = os.path.join(EXPERIMENTS_PATH, experiment)

        losses_file = open(os.path.join(path, LOSSES_FILE), "r")
        losses_strings = np.array(losses_file.read().splitlines())
        losses = losses_strings.astype(np.float)
        info_file = open(os.path.join(path, INFO_FILE), "r")
        title = info_file.readline()
        plot_losses(losses, title, path)
    return 5


if __name__ == "__main__":
    main()
