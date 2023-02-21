import random
import numpy as np
import matplotlib.pyplot as plt


class Bandit:
    def __init__(self, true_mean):
        self.true_mean = true_mean
        self.estimated_mean = 0
        self.num_pulls = 0

    def pull(self):
        return random.gauss(self.true_mean, 1)

    def update(self, x):
        self.num_pulls += 1
        self.estimated_mean = (1 - 1 / self.num_pulls) * self.estimated_mean + 1 / self.num_pulls * x


def run_experiment(m1, m2, m3, eps, N):
    bandits = [Bandit(m1), Bandit(m2), Bandit(m3)]

    data = []
    for i in range(N):
        p = random.random()
        if p < eps:
            j = random.choice(range(3))
        else:
            j = max(range(3), key=lambda x: bandits[x].estimated_mean)

        x = bandits[j].pull()
        bandits[j].update(x)

        data.append(x)

    cumulative_average = np.cumsum(data) / (np.arange(N) + 1)

    plt.plot(cumulative_average)
    plt.plot(np.ones(N) * m1)
    plt.plot(np.ones(N) * m2)
    plt.plot(np.ones(N) * m3)
    plt.xscale('log')
    plt.show()


random.seed(0)
run_experiment(1.0, 2.0, 3.0, 0.1, 100000)
