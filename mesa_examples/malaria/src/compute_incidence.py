import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

class compute_incidence():
    """Computing incidence"""

    def __init__(self, model):

        # Assign parameters
        self.year_init = model.year_init
        self.year_termi = model.year_termi
        self.day_init = model.day_init
        self.dt = model.dt

        # Solution
        t_init = model.tau * model.n_step_per_day
        n_step = t_init + len(model.tspan)
        self.S_h = np.zeros([n_step])
        self.E_h = np.zeros([n_step])
        self.I_h = np.zeros([n_step])
        self.T_h = np.zeros([n_step])
        self.A = np.zeros([n_step])
        self.S_v = np.zeros([n_step])
        self.E_v = np.zeros([n_step])
        self.I_v = np.zeros([n_step])

        # Loop over human agents
        for j in range(model.n_agent_h):

            human_id = model.schedule.agents[j].human_id
            state_j = human_id % 10

            if state_j == 0:
                self.S_h += model.schedule.agents[j].sol
            elif state_j == 1:
                self.E_h += model.schedule.agents[j].sol
            elif state_j == 2:
                self.I_h += model.schedule.agents[j].sol
            elif state_j == 3:
                self.T_h += model.schedule.agents[j].sol

        # Loop over mosquito agents
        for j in model.unique_id_v:

            vector_id = model.schedule.agents[j].vector_id
            state_j = vector_id % 10

            if state_j == 0:
                self.A += model.schedule.agents[j].sol
            elif state_j == 1:
                self.S_v += model.schedule.agents[j].sol
            elif state_j == 2:
                self.E_v += model.schedule.agents[j].sol
            elif state_j == 3:
                self.I_v += model.schedule.agents[j].sol

        self.b_history0 = model.b_history0
        self.mu_a_history0 = model.mu_a_history0
        self.delta_a_history0 = model.delta_a_history0
        self.mu_v_history0 = model.mu_v_history0
        self.delta_v_history0 = model.delta_v_history0

        self.n_step_per_day = model.n_step_per_day
        self.tau = model.tau

        # self.result_path = model.result_path
        # self.suffix = model.suffix

        self.b = model.b_history

        self.N_h = self.S_h + self.E_h + self.I_h + self.T_h
        self.N_v = self.S_v + self.E_v + self.I_v

        self.n_year = self.year_termi - self.year_init + 1
        self.n_step_delay = model.tau * model.n_step_per_day
        self.n_step = len(model.tspan)

        # Initialize
        incidence_short = np.zeros([self.n_step])
        incidence_long = np.zeros([self.n_step])
        incidence_relapse = np.zeros([self.n_step])

        for i in range(self.n_step):
            index_short = i + (self.n_step_delay - model.tau_s * model.n_step_per_day)
            index_long = i + (self.n_step_delay - model.tau_l * model.n_step_per_day)
            index_relapse = i + (self.n_step_delay - model.tau_r * model.n_step_per_day)

            incidence_short[i] = (model.p * self.b[index_short] * model.beta_hv * self.I_v[index_short]
                                  / self.N_h[index_short] * self.S_h[index_short] * np.exp(-model.delta_h * model.tau_s))
            incidence_long[i] = ((1 - model.p) * self.b[index_long] * model.beta_hv * self.I_v[index_long]
                                 / self.N_h[index_long] * self.S_h[index_long] * np.exp(-model.delta_h * model.tau_l))
            incidence_relapse[i] = model.q * model.rho_h * self.T_h[index_relapse] * np.exp(-model.delta_h * model.tau_r)

        incidence_total = incidence_short + incidence_long + incidence_relapse

        incidence = np.zeros([self.n_step -1, 4])
        incidence[:, 0] = (incidence_short[:-1] + incidence_short[1:]) * model.dt / 2
        incidence[:, 1] = (incidence_long[:-1] + incidence_long[1:]) * model.dt / 2
        incidence[:, 2] = (incidence_relapse[:-1] + incidence_relapse[1:]) * model.dt / 2
        incidence[:, 3] = (incidence_total[:-1] + incidence_total[1:]) * model.dt / 2

        self.incidence = incidence

        # Compute daily incidence
        mat4day = np.kron(np.eye(int(self.n_step * self.dt)), np.ones((1, int(1 / self.dt))))
        incidence_day = np.matmul(mat4day, self.incidence)

        self.incidence_day = incidence_day

        # Compute weekly incidence
        tspan_day = np.arange(self.day_init, 365 * self.n_year + 1)
        day_point = np.append([0], np.arange(244, len(tspan_day) + 1, 365))

        incidence_week = np.empty((0, 4))
        for i in range(self.n_year):
            incidence_day_temp = incidence_day[day_point[i]:day_point[i + 1], :]
            remainder = np.shape(incidence_day_temp)[0] % 7
            n_week = int(np.shape(incidence_day_temp)[0] / 7)
            mat4week = np.kron(np.eye(n_week), np.ones((1, 7)))
            incidence_week = np.append(incidence_week, np.matmul(mat4week, incidence_day_temp[:-remainder, :]), axis=0)
            incidence_week[-n_week:, :] += np.sum(incidence_day_temp[-remainder:, :], 0) / n_week

        self.incidence_week = incidence_week

        # Compute annual incidence
        incidence_year = np.zeros([self.n_year, 4])
        for i in range(self.n_year):
            incidence_year[i, :] = sum(self.incidence_day[day_point[i]:day_point[i + 1], :])

        self.incidence_year = incidence_year

    def plot_incidence(self):

        # Load incidence data
        incidence_data = pd.read_csv('../data/case_region_week.csv')
        incidence_data_init = incidence_data[incidence_data['연도'] > self.year_init]

        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        t_init4plot = (self.tau + (365 - self.day_init)) * self.n_step_per_day

        fig, ax1 = plt.subplots()
        ax1.plot(self.N_h[t_init4plot:], color=colors[0])
        ax1.plot(self.S_h[t_init4plot:], color=colors[1])
        ax2 = ax1.twinx()
        ax2.plot(self.E_h[t_init4plot:], color=colors[2])
        ax2.plot(self.I_h[t_init4plot:], color=colors[3])
        ax2.plot(self.T_h[t_init4plot:], color=colors[4])
        ax1.set_ylim([0, 7e+6])
        ax1.legend(['Total', 'Susceptible'], loc='upper left')
        ax2.legend(['Exposed', 'Infectious', 'Treated'], loc='upper right')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Population')
        ax2.set_ylabel('Population')
        ax1.set_title('Human')
        # plt.savefig('{}/human{}.eps'.format(self.result_path, self.suffix), format='eps')
        plt.show()

        fig, ax1 = plt.subplots()
        ax1.plot(self.A[t_init4plot:], color=colors[0])
        ax1.plot(self.N_v[t_init4plot:], color=colors[1])
        ax1.plot(self.S_v[t_init4plot:], color=colors[2])
        ax2 = ax1.twinx()
        ax2.plot(self.E_v[t_init4plot:], color=colors[3])
        ax2.plot(self.I_v[t_init4plot:], color=colors[4])
        ax1.set_ylim([0, 2e+9])
        ax2.set_ylim([0, 400])
        ax1.legend(['Immature', 'Adult total', 'Susceptible'], loc='upper left')
        ax2.legend(['Exposed', 'Infectious'], loc='upper right')
        ax1.set_xlabel('Day')
        ax1.set_ylabel('Population')
        ax2.set_ylabel('Population')
        ax1.set_title('Mosquito')
        # plt.savefig('{}/mosquito{}.eps'.format(self.result_path, self.suffix), format='eps')
        plt.show()

        fig, ax = plt.subplots()
        incidence_total = incidence_data_init.groupby(by=['연도', '주'], as_index=False).sum()
        ax.plot(incidence_total['발생수'], '*:', linewidth=0.5, markersize=3)
        ax.plot(self.incidence_week[34:, 0], '-', linewidth=0.5)
        ax.plot(self.incidence_week[34:, 1], '-', linewidth=0.5)
        ax.plot(self.incidence_week[34:, 2], '-', linewidth=0.5)
        ax.plot(self.incidence_week[34:, 3], 'o-', fillstyle='none', linewidth=0.5, markersize=4)
        ax.legend(['Data', 'Short', 'Long', 'Relapse', 'Total'], loc='upper left')
        ax.set_xlabel('Week')
        ax.set_ylabel('Cases')
        # plt.savefig('{}/incidence_week{}.eps'.format(self.result_path, self.suffix), format='eps')
        plt.show()

        fig, ax = plt.subplots()
        incidence_total = incidence_data_init.groupby(by=['연도'], as_index=False).sum()
        ax.plot(incidence_total['발생수'], '*:')
        ax.plot(self.incidence_year[1:, 3], 'o-', fillstyle='none')
        ax.legend(['Data', 'Total'], loc='upper left')
        ax.set_xlabel('Year')
        ax.set_ylabel('Cases')
        # plt.savefig('{}/incidence_year{}.eps'.format(self.result_path, self.suffix), format='eps')
        plt.show()

        # Save to csv
        df = pd.DataFrame(np.concatenate(([self.b_history0], [self.mu_a_history0], [self.delta_a_history0],
                                          [self.mu_v_history0], [self.delta_v_history0])).T,
                          columns=['b', 'mu_a', 'delta_a', 'mu_v', 'delta_v'])
        # df.to_csv('{}/temp_param{}.csv'.format(self.result_path, self.suffix), index=False)

        # Save to csv
        df = pd.DataFrame(np.concatenate(([self.S_h], [self.E_h], [self.I_h], [self.T_h], [self.A], [self.S_v],
                                          [self.E_v], [self.I_v])).T, columns=['S_h', 'E_h', 'I_h', 'T_h', 'A',
                                                                               'S_v', 'E_v', 'I_v'])
        # df.to_csv('{}/state{}.csv'.format(self.result_path, self.suffix), index=False)