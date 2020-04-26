import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

from DP import DP
from FQI import FQI


data = np.loadtxt(open('data_test.csv', 'rb'), delimiter=",", skiprows=0)
config = {'S0': 100,
          'K': 105,
          'mu': 0.05,
          'sigma': 0.15,
          'r': 0.03,
          'M': 1,
          'T': 24,
          'n_mc': 10000,
          'risk_lambda': 0.001}

dp = DP(config)
x = dp.s_to_x(data)
variables = dp.initialization(data, x)
dp_results = dp.price(data, x, var_dict=variables)

fqi = FQI(config)

all_eta = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
q_values = {}

for i, eta in enumerate(all_eta):
    print('Evaluating option price on noise level:', eta)
    action_off, reward_off = fqi.make_off_policy_data(dp_results=dp_results, eta=eta)
    q_value = fqi.price(data, x, action_off, reward_off, variables)
    q_values[i] = q_value


def plot_fqi(q_values, all_eta):
    def q_plot(i, ax):
        ax.plot(q_values[i][dp.plot_idx, :].T)
        ax.set_xlabel('Time Steps')
        ax.set_title('Noise Level: ' + str(all_eta[i]))

    fig, axs = plt.subplots(nrows=3, ncols=2, constrained_layout=True)

    for i, ax in enumerate(axs.flat):
        q_plot(i, ax)
    plt.show()


plot_fqi(q_values, all_eta)

option_value = []
for i in range(len(all_eta)):
    option = -np.mean(q_values[i][:, 0])
    option_value.append(option)

plt.figure()
plt.plot(all_eta, option_value, 'ro')
plt.xlabel('Noise Level')
plt.ylabel('Option Price')
plt.title('Option Price in Different Noise Levels')
plt.show()
