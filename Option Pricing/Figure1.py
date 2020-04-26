import numpy as np
import matplotlib.pyplot as plt

from DP import DP


data = np.loadtxt(open('data_test.csv', 'rb'), delimiter=",", skiprows=0)
config = {'S0': 100,
          'K': 105,
          'mu': 0.05,
          'sigma': 0.15,
          'r': 0.03,
          'M': 1,
          'T': 24,
          'n_mc': 10000,
          'risk_lambda': 0}

dp = DP(config)
x = dp.s_to_x(data)
variables = dp.initialization(data, x)
dp_results = dp.price(data, x, var_dict=variables)


def plot_dp(data, x, dp_results):
    action = dp_results['action']
    pi = dp_results['pi']
    reward = dp_results['reward']
    q = dp_results['q']

    x_labels = {'Stock Price': data, 'State Variable': x, 'Action': action,
                'Porfolio Value': pi, 'Reward': reward, 'Q-Value': q}

    def var_plot(i, ax):
        title = list(x_labels.keys())[i]
        var = x_labels[title]
        ax.plot(var[dp.plot_idx, :].T)
        ax.set_xlabel('Time Steps')
        ax.set_title(title)

    fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True)

    for i, ax in enumerate(axs.flat):
        var_plot(i, ax)
    plt.show()

plot_dp(data, x, dp_results)

print('The mean of Portfolio Value:', dp_results['pi'][:, 0].mean())
print('The mean of Q-Value:', dp_results['q'][:, 0].mean())
