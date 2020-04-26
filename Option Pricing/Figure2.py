import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

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


def bs_call(t=0, S0=config['S0'], K=config['K'], r=config['r'], sigma=config['sigma'], T=config['M']):
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma ** 2) * (T - t)) / sigma / np.sqrt(T - t)
    d2 = (np.log(S0 / K) + (r - 0.5 * sigma ** 2) * (T - t)) / sigma / np.sqrt(T - t)
    price = S0 * norm.cdf(d1) - K * np.exp(-r * (T - t)) * norm.cdf(d2)
    return price


bs = bs_call()

all_lambda = list(np.arange(11) * 0.001)
results = []

for i in range(len(all_lambda)):
    print('Evaluating option price of risk lambda:', all_lambda[i])
    config['risk_lambda'] = all_lambda[i]
    dp = DP(config)
    x = dp.s_to_x(data)
    variables = dp.initialization(data, x)
    dp_results = dp.price(data, x, var_dict=variables)
    results.append(-np.mean(dp_results['q'][:, 0]))

plt.figure()
plt.hlines(y=bs, xmin=0, xmax=0.01, abel='BS Price: '+str(bs))
plt.plot(all_lambda, results, 'ro')
plt.xlabel('Risk Lambda')
plt.ylabel('Option Price')
plt.show()

