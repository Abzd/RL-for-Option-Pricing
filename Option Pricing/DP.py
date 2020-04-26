import numpy as np
import matplotlib.pyplot as plt
import bspline
import bspline.splinelab as splinelab


class DP:
    def __init__(self, config):
        self.config = config

        self.K = self.config['K']
        self.mu = self.config['mu']
        self.sigma = self.config['sigma']
        self.r = self.config['r']
        self.T = self.config['T']
        self.M = self.config['M']
        self.n_mc = self.config['n_mc']
        self.risk_lambda = self.config['risk_lambda']

        self.plot_path = 10
        self.num_basis = 12

        step_size = self.n_mc // self.plot_path
        self.plot_idx = np.arange(step_size, self.n_mc, step_size)
        self.dt = self.M / self.T
        self.gamma = np.exp(-self.r * self.dt)

    def s_to_x(self, data):
        """
        Convert S to X.
        :return: State variable X of shape ()
        """
        x = -(self.mu - self.sigma**2 / 2) * np.arange(self.T + 1) * self.dt + np.log(data)

        assert x.shape == (self.n_mc, self.T + 1), 'Error: shape of X'
        return x

    def initialization(self, data, x):
        """
        Feed in data and initialize variables needed in DP solution.
        :return: Dictionary of variables.
        """
        var_dict = {}

        var_dict['ds'] = data[:, 1:self.T+1] - 1 / self.gamma * data[:, 0:self.T]
        var_dict['ds_hat'] = var_dict['ds'] - np.mean(var_dict['ds'], axis=0)

        var_dict['pi'] = np.zeros((self.n_mc, self.T + 1))
        var_dict['pi_hat'] = np.zeros_like(var_dict['pi'])
        var_dict['action'] = np.zeros_like(var_dict['pi'])
        var_dict['q'] = np.zeros_like(var_dict['pi'])
        var_dict['reward'] = np.zeros_like(var_dict['pi'])

        var_dict['pi'][:, self.T] = np.maximum(data[:, self.T] - self.K, 0)
        var_dict['pi_hat'][:, self.T] = var_dict['pi'][:, self.T] - np.mean(var_dict['pi'][:, self.T])
        var_dict['action'][:, self.T] = 0
        var_dict['q'][:, self.T] = -var_dict['pi'][:, self.T] - self.risk_lambda * np.var(var_dict['pi'][:, self.T])
        var_dict['reward'][:, self.T] = -self.risk_lambda * np.var(var_dict['pi'][:, self.T])

        x_min, x_max = np.min(x), np.max(x)
        tau = np.linspace(x_min, x_max, self.num_basis)
        k = splinelab.aptknt(tau=tau, order=3)
        basis = bspline.Bspline(k, order=3)
        var_dict['func_x'] = np.zeros((self.n_mc, self.T + 1, self.num_basis))
        for t in range(self.T + 1):
            xt = x[:, t]
            var_dict['func_x'][:, t, :] = np.array([basis(element) for element in xt])

        print('The shape of pi / action / q:', var_dict['pi'].shape)
        print('The shape of func_x:', var_dict['func_x'].shape)

        return var_dict

    def compute_a(self, t, func_x, ds_hat, reg_param=1e-3):
        x_mat = func_x[:, t, :]
        ds_hat_t2 = (ds_hat[:, t]**2).reshape(self.n_mc, 1)
        a_mat = np.dot(x_mat.T, x_mat * ds_hat_t2) + reg_param * np.eye(self.num_basis)

        assert a_mat.shape == (self.num_basis, self.num_basis), 'Wrong with the shape of a_mat.'
        return a_mat

    def compute_b(self, t, func_x, ds_hat, pi_hat, ds):
        x_mat = func_x[:, t, :]
        # coe = 1 / (2 * self.gamma * self.risk_lambda)
        coe = 0
        temp = pi_hat[:, t + 1] * ds_hat[:, t] + coe * ds[:, t]
        b_mat = np.dot(x_mat.T, temp)

        assert b_mat.shape == (self.num_basis,), 'Wrong with the shape of b_mat.'
        return b_mat

    def compute_c(self, t, func_x, reg_param=1e-3):
        x_mat = func_x[:, t, :]
        c_mat = np.dot(x_mat.T, x_mat) + reg_param * np.eye(self.num_basis)

        assert c_mat.shape == (self.num_basis, self.num_basis), 'Wrong with the shape of c_mat.'
        return c_mat

    def compute_d(self, t, func_x, reward, q):
        x_mat = func_x[:, t, :]
        temp = reward[:, t] + self.gamma * q[:, t + 1]
        d_mat = np.dot(x_mat.T, temp)

        assert d_mat.shape == (self.num_basis,), 'Wrong with the shape of d_mat.'
        return d_mat

    def price(self, data, x, var_dict):
        """
        Pricing the option by dp solution.
        """
        ds = var_dict['ds']
        ds_hat = var_dict['ds_hat']
        pi = var_dict['pi']
        pi_hat = var_dict['pi_hat']
        reward = var_dict['reward']
        action = var_dict['action']
        q = var_dict['q']
        func_x = var_dict['func_x']

        for t in reversed(range(self.T)):
            a_mat = self.compute_a(t, func_x, ds_hat)
            b_mat = self.compute_b(t, func_x, ds_hat, pi_hat, ds)

            phi = np.dot(np.linalg.inv(a_mat), b_mat).reshape(self.num_basis, 1)
            x_mat = func_x[:, t, :].reshape(self.n_mc, self.num_basis)
            action[:, t] = np.dot(x_mat, phi).reshape(self.n_mc)

            pi[:, t] = self.gamma * (pi[:, t + 1] - action[:, t] * ds[:, t])
            pi_hat[:, t] = pi[:, t] - np.mean(pi[:, t])
            reward[:, t] = self.gamma * action[:, t] * ds[:, t] - self.risk_lambda * np.var(pi[:, t])

            c_mat = self.compute_c(t, func_x)
            d_mat = self.compute_d(t, func_x, reward, q)
            omega = np.dot(np.linalg.inv(c_mat), d_mat)
            q[:, t] = np.dot(x_mat, omega)

            dp_results = {'action': action, 'pi': pi,
                          'reward': reward, 'q': q,
                          'ds': ds, 'ds_hat': ds_hat,
                          'pi_hat': pi_hat}
        return dp_results


if __name__ == '__main__':
    config_test = {'S0': 100,
                   'K': 100,
                   'mu': 0.05,
                   'sigma': 0.15,
                   'r': 0.03,
                   'M': 1,
                   'T': 24,
                   'n_mc': 10000,
                   'risk_lambda': 0.001}
    data_test = np.loadtxt(open('data_test.csv', 'rb'), delimiter=",", skiprows=0)

    dp = DP(config_test)
    x = dp.s_to_x(data_test)
    variables = dp.initialization(data_test, x)
    dp.price(data_test, x, var_dict=variables)

