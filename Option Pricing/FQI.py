import numpy as np
import matplotlib.pyplot as plt
from DP import DP


class FQI(DP):
    def make_off_policy_data(self, dp_results, eta):
        action = dp_results['action']
        reward = dp_results['reward']
        pi = dp_results['pi']
        pi_hat = dp_results['pi_hat']
        ds = dp_results['ds']

        action_off = action.copy()
        reward_off = reward.copy()
        pi_off = pi.copy()
        pi_hat_off = pi_hat.copy()

        np.random.seed(40)

        for t in reversed(range(self.T)):
            action_off[:, t] *= np.random.uniform(1 - eta, 1 + eta, self.n_mc)
            pi_off[:, t] = self.gamma * (pi_off[:, t + 1] - action_off[:, t] * ds[:, t])
            pi_hat_off[:, t] = pi_off[:, t] - np.mean(pi_off[:, t])
            reward_off[:, t] = self.gamma * action_off[:, t] * ds[:, t] - self.risk_lambda * np.var(pi_off[:, t])

        return action_off, reward_off

    def compute_psi2(self, func_x, action_off):
        action_1 = np.expand_dims(action_off, axis=0)
        action_2 = action_1 ** 2 / 2
        ones = np.ones((1, self.n_mc, self.T + 1))
        action_stack = np.vstack((ones, action_1, action_2))
        func_x_swap = np.swapaxes(np.swapaxes(func_x, 0, 2), 1, 2)

        action_temp = np.expand_dims(action_stack, axis=1)
        func_x_swap = np.expand_dims(func_x_swap, axis=0)
        psi_mat = np.multiply(action_temp, func_x_swap)
        psi_mat = psi_mat.reshape((-1, self.n_mc, self.T + 1), order='F')

        psi_1 = np.expand_dims(psi_mat, axis=1)
        psi_2 = np.expand_dims(psi_mat, axis=0)
        psi2 = np.sum(np.multiply(psi_1, psi_2), axis=2)

        print('The shape of psi2:', psi2.shape)
        return psi2, psi_mat

    def compute_s(self, t, psi2, reg_param=1e-3):
        s_mat_t = psi2[:, :, t]
        dim_reg = s_mat_t.shape[0]
        s_mat_t += reg_param * np.eye(dim_reg)
        return s_mat_t

    def compute_m(self, t, psi_mat, reward, q_star):
        temp = reward[:, t] + self.gamma * q_star[:, t + 1]
        m_mat = np.dot(psi_mat[:, :, t], temp)
        return m_mat

    def price(self, data, x, action_off, reward_off, var_dict):
        """
        Pricing the option using RL method.
        """
        ds = var_dict['ds']
        ds_hat = var_dict['ds_hat']
        q = var_dict['q']
        func_x = var_dict['func_x']

        q_star = np.zeros_like(q)
        q_star[:, self.T] = q[:, self.T]
        max_q = np.zeros_like(q)
        max_q[:, self.T] = q[:, self.T]

        pi = np.zeros_like(q)
        pi_hat = np.zeros_like(q)
        pi[:, self.T] = var_dict['pi'][:, self.T]
        pi_hat[:, self.T] = var_dict['pi_hat'][:, self.T]
        action_star = np.zeros_like(q)
        action_star[:, self.T] = 0

        psi2, psi_mat = self.compute_psi2(func_x, action_off)

        # Helper function to compute max_q
        def get_max_q(u, action_t):
            u0, u1, u2 = u[0, :], u[1, :], u[2, :]
            max_q = u0 + u1 * action_t + u2 * (action_t**2) / 2
            return max_q

        for t in reversed(range(self.T)):
            s_mat = self.compute_s(t, psi2)
            m_mat = self.compute_m(t, psi_mat, reward_off, q_star)
            w_mat = np.dot(np.linalg.inv(s_mat), m_mat)
            w_mat = w_mat.reshape((3, self.num_basis), order='F')
            u_mat = np.dot(w_mat, func_x[:, t, :].T)

            a_mat = self.compute_a(t, func_x, ds_hat)
            b_mat = self.compute_b(t, func_x, ds_hat, pi_hat, ds)
            
            phi = np.dot(np.linalg.inv(a_mat), b_mat).reshape(self.num_basis, 1)
            x_mat = func_x[:, t, :].reshape(self.n_mc, self.num_basis)
            action_star[:, t] = np.dot(x_mat, phi).reshape(self.n_mc)

            pi[:, t] = self.gamma * (pi[:, t + 1] - action_star[:, t] * ds[:, t])
            pi_hat[:, t] = pi[:, t] - np.mean(pi[:, t])

            max_q[:, t] = get_max_q(u=u_mat, action_t=action_star[:, t])
            q_star[:, t] = max_q[:, t]

        return q_star

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
    dp_results = dp.price(data_test, x, var_dict=variables)

    fqi = FQI(config_test)
    action_off, reward_off = fqi.make_off_policy_data(dp_results=dp_results)
    qq = fqi.price(data_test, x, action_off, reward_off, variables)
