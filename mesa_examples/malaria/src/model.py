from mesa import Agent, Model
from mesa.time import SimultaneousActivation
from itertools import product
import numpy as np


class HumanAgent(Agent):
    """Human Agent"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Assign characteristics
        self.characteristics = self.model.combination_h[unique_id]
        self.region = self.characteristics[0]
        self.state = self.characteristics[1]

        # Set initial condition
        if self.state == 0:
            self.history = np.ones(self.model.tau * self.model.n_step_per_day) * self.model.N_h0 / self.model.n_region
            self.sol0 = self.model.N_h0 / self.model.n_region
        else:
            self.history = np.zeros(self.model.tau * self.model.n_step_per_day)
            self.sol0 = 0

        # Solution
        self.sol = np.append(self.history, self.sol0)

        # Initialize agent level parameters
        self.increment = 0
        self.sol_next = 0

    def step_S_h(self):

        S_h_ = self.sol[-1]

        E_h_id = self.human_id + 1
        E_h_ = self.model.schedule.agents[self.model.mapping_h[E_h_id]].sol[-1]

        I_h_id = self.human_id + 2
        I_h_ = self.model.schedule.agents[self.model.mapping_h[I_h_id]].sol[-1]

        T_h_id = self.human_id + 3
        T_h_ = self.model.schedule.agents[self.model.mapping_h[T_h_id]].sol[-1]

        N_h_ = S_h_ + E_h_ + I_h_ + T_h_

        lambda_h_ = self.model.lambda_h_history[-1]

        self.increment = (self.model.mu_h * N_h_ - lambda_h_ * S_h_
                         + (1 - self.model.q) * self.model.rho_h * T_h_ - self.model.delta_h * S_h_) * self.model.dt

        self.sol_next = S_h_ + self.increment

    def step_E_h(self):

        S_h_id = self.human_id - 1
        S_h_ = self.model.schedule.agents[self.model.mapping_h[S_h_id]].sol[-1]
        S_h_tau_s_ = self.model.schedule.agents[self.model.mapping_h[S_h_id]].sol[-(self.model.tau_s * self.model.n_step_per_day + 1)]
        S_h_tau_l_ = self.model.schedule.agents[self.model.mapping_h[S_h_id]].sol[-(self.model.tau_l * self.model.n_step_per_day + 1)]

        E_h_ = self.sol[-1]

        T_h_id = self.human_id + 2
        T_h_ = self.model.schedule.agents[self.model.mapping_h[T_h_id]].sol[-1]
        T_h_tau_r_ = self.model.schedule.agents[self.model.mapping_h[T_h_id]].sol[-(self.model.tau_r * self.model.n_step_per_day + 1)]

        lambda_h_ = self.model.lambda_h_history[-1]
        lambda_h_tau_s_ = self.model.lambda_h_history[-(self.model.tau_s * self.model.n_step_per_day + 1)]
        lambda_h_tau_l_ = self.model.lambda_h_history[-(self.model.tau_l * self.model.n_step_per_day + 1)]

        self.increment = (lambda_h_ * S_h_
                          - self.model.p * lambda_h_tau_s_ * S_h_tau_s_ * np.exp(-self.model.delta_h * self.model.tau_s)
                          - (1 - self.model.p) * lambda_h_tau_l_ * S_h_tau_l_ * np.exp(-self.model.delta_h * self.model.tau_l)
                          + self.model.q * self.model.rho_h * T_h_
                          - self.model.q * self.model.rho_h * T_h_tau_r_ * np.exp(-self.model.delta_h * self.model.tau_r)
                          - self.model.delta_h * E_h_) * self.model.dt

        self.sol_next = E_h_ + self.increment

    def step_I_h(self):

        S_h_id = self.human_id - 2
        S_h_tau_s_ = self.model.schedule.agents[self.model.mapping_h[S_h_id]].sol[-(self.model.tau_s * self.model.n_step_per_day + 1)]
        S_h_tau_l_ = self.model.schedule.agents[self.model.mapping_h[S_h_id]].sol[-(self.model.tau_l * self.model.n_step_per_day + 1)]

        I_h_ = self.sol[-1]

        T_h_id = self.human_id + 1
        T_h_tau_r_ = self.model.schedule.agents[self.model.mapping_h[T_h_id]].sol[-(self.model.tau_r * self.model.n_step_per_day + 1)]

        lambda_h_tau_s_ = self.model.lambda_h_history[-(self.model.tau_s * self.model.n_step_per_day + 1)]
        lambda_h_tau_l_ = self.model.lambda_h_history[-(self.model.tau_l * self.model.n_step_per_day + 1)]

        self.increment = (self.model.p * lambda_h_tau_s_ * S_h_tau_s_ * np.exp(-self.model.delta_h * self.model.tau_s)
                          + (1 - self.model.p) * lambda_h_tau_l_ * S_h_tau_l_ * np.exp(-self.model.delta_h * self.model.tau_l)
                          + self.model.q * self.model.rho_h * T_h_tau_r_ * np.exp(-self.model.delta_h * self.model.tau_r)
                          - self.model.gamma_h * I_h_ - self.model.delta_h * I_h_) * self.model.dt

        self.sol_next = I_h_ + self.increment

    def step_T_h(self):

        I_h_id = self.human_id - 1
        I_h_ = self.model.schedule.agents[self.model.mapping_h[I_h_id]].sol[-1]
        T_h_ = self.sol[-1]

        self.increment = (self.model.gamma_h * I_h_ - self.model.rho_h * T_h_
                          - self.model.delta_h * T_h_) * self.model.dt

        self.sol_next = T_h_ + self.increment

    def step(self):

        if self.state == 0:
            self.step_S_h()
        elif self.state == 1:
            self.step_E_h()
        elif self.state == 2:
            self.step_I_h()
        elif self.state == 3:
            self.step_T_h()

        if self.sol_next < 0:
            self.sol_next = 0

    def advance(self):

        self.sol = np.append(self.sol, self.sol_next)


class MosquitoAgent(Agent):
    """Mosquito Agent"""

    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

        # Assign characteristics
        self.index4v = unique_id - self.model.n_agent_h
        self.characteristics = self.model.combination_v[self.index4v]
        self.region = self.characteristics[0]
        self.state = self.characteristics[1]

        # Set initial condition
        if self.state == 1:
            self.history = np.zeros(self.model.tau * self.model.n_step_per_day)
            self.sol0 = self.model.N_h0 / 20 / self.model.n_region
        elif self.state == 3:
            self.history = np.zeros(self.model.tau * self.model.n_step_per_day)
            self.sol0 = 100 / self.model.n_region
        else:
            self.history = np.zeros(self.model.tau * self.model.n_step_per_day)
            self.sol0 = 0

        # Solution
        self.sol = np.append(self.history, self.sol0)

        # Initialize agent level parameters
        self.increment = 0
        self.sol_next = 0

    def step_A(self):

        if self.model.year_index == 1:
            k_a_ = self.model.c_init * self.model.N_h0 / self.model.n_region
        else:
            k_a_ = self.model.c * self.model.N_h0 / self.model.n_region

        A_ = self.sol[-1]

        S_v_id = self.vector_id + 1
        S_v_supplied = self.model.schedule.agents[self.model.mapping_v[S_v_id]].S_v_supplied

        E_v_id = self.vector_id + 2
        E_v_ = self.model.schedule.agents[self.model.mapping_v[E_v_id]].sol[-1]

        I_v_id = self.vector_id + 3
        I_v_ = self.model.schedule.agents[self.model.mapping_v[I_v_id]].sol[-1]

        N_v_ = S_v_supplied + E_v_ + I_v_

        self.increment = (self.model.mu_a_ * (1 - A_ / k_a_) * N_v_
                          - self.model.mu_v_ * A_ - self.model.delta_a_ * A_) * self.model.dt

        self.sol_next = A_ + self.increment

    def step_S_v(self):

        A_id = self.vector_id - 1
        A_ = self.model.schedule.agents[self.model.mapping_v[A_id]].sol[-1]

        S_v_ = self.sol[-1]
        S_v_supplied = self.S_v_supplied

        self.increment = (self.model.mu_v_ * A_ - self.model.lambda_v_ * S_v_supplied
                          - self.model.delta_v_ * S_v_supplied) * self.model.dt

        self.sol_next = S_v_ + self.increment

    def step_E_v(self):

        S_v_id = self.vector_id - 1
        S_v_supplied = self.model.schedule.agents[self.model.mapping_v[S_v_id]].S_v_supplied

        E_v_ = self.sol[-1]

        self.increment = (self.model.lambda_v_ * S_v_supplied - self.model.nu_v * E_v_
                          - self.model.delta_v_ * E_v_) * self.model.dt

        self.sol_next = E_v_ + self.increment

    def step_I_v(self):

        E_v_id = self.vector_id - 1
        E_v_ = self.model.schedule.agents[self.model.mapping_v[E_v_id]].sol[-1]

        I_v_ = self.sol[-1]

        self.increment = (self.model.nu_v * E_v_ - self.model.delta_v_ * I_v_) * self.model.dt

        self.sol_next = I_v_ + self.increment

    def step(self):

        if self.state == 0:
            self.step_A()
        elif self.state == 1:
            self.step_S_v()
        elif self.state == 2:
            self.step_E_v()
        elif self.state == 3:
            self.step_I_v()

        if self.sol_next < 0:
            self.sol_next = 0

    def advance(self):

        self.sol = np.append(self.sol, self.sol_next)


class MalariaModel(Model):
    """Population Level"""

    def __init__(self, params):

        # Assign parameters
        self.region_h = params['region_h']
        self.state_h = params['state_h']
        self.region_v = params['region_v']
        self.state_v = params['state_v']

        self.year_init = params['year_init']
        self.year_termi = params['year_termi']
        self.day_init = params['day_init']
        self.tspan = params['tspan']
        self.tspan_vsupply = params['tspan_vsupply']
        self.dt = params['dt']

        self.N_h0 = params['N_h0']

        self.mu_h = params['mu_h']
        self.delta_h = params['delta_h']
        self.c_init = params['c_init']
        self.c = params['c']
        self.p = params['p']
        self.beta_hv = params['beta_hv']
        self.beta_vh = params['beta_vh']
        self.tau_s = params['tau_s']
        self.tau_l = params['tau_l']
        self.tau_r = params['tau_r']
        self.gamma_h = params['gamma_h']
        self.q = params['q']
        self.rho_h = params['rho_h']
        self.nu_v = params['nu_v']

        self.b = params['b']
        self.mu_a = params['mu_a']
        self.delta_a = params['delta_a']
        self.mu_v = params['mu_v']
        self.delta_v = params['delta_v']

        # self.result_path = params['result_path']
        # self.suffix = params['suffix']

        # Number of each characteristics
        self.n_region = len(self.region_h)

        # Maximum delay
        self.tau = max(self.tau_s, self.tau_l, self.tau_r)

        # Steps per day
        self.n_step_per_day = int(1 / self.dt)

        # Set scheduler
        self.schedule = SimultaneousActivation(self)

        # Types of state
        self.n_agent_v = len(self.state_v)

        # Generate human agent combination
        self.combination_h = list(product(self.region_h, self.state_h))
        self.n_agent_h = len(self.combination_h)

        # Initialize human agents
        human_id_all = []# (region, state) unique_id cannot distinguish agent's characteristics
        for i in range(self.n_agent_h):
            agent = HumanAgent(i, self)
            agent.human_id = self.combination_h[i][0] * 100 + self.combination_h[i][1]
            human_id_all.append(agent.human_id)
            self.schedule.add(agent)

        # Mapping human_id -> unique_id
        self.mapping_h = dict(zip(human_id_all, np.arange(0, self.n_agent_h)))

        # Generate mosquito agent combination
        self.combination_v = list(product(self.region_v, self.state_v))
        self.n_agent_v = len(self.combination_v)
        self.unique_id_v = np.arange(self.n_agent_h, self.n_agent_h + self.n_agent_v)

        # Initialize mosquito agents
        vector_id_all = []
        for i in self.unique_id_v:
            agent = MosquitoAgent(i, self)
            agent.vector_id = self.combination_v[agent.index4v][0] * 100 + self.combination_v[agent.index4v][1]
            vector_id_all.append(agent.vector_id)
            self.schedule.add(agent)

        # Mapping vector_id -> unique_id
        self.mapping_v = dict(zip(vector_id_all, self.unique_id_v))

        # b history
        b_history = np.zeros(self.tau * self.n_step_per_day)
        for i in range(self.day_init):
            for j in range(self.n_step_per_day):
                b_history[self.tau * self.n_step_per_day - 1 - (i * self.n_step_per_day) - j] = self.b[
                    self.day_init - 1 - i]
        self.b_history = b_history

        # I_v history
        I_v_history = np.zeros(self.tau * self.n_step_per_day)
        for i in self.unique_id_v:
            vector_id = self.schedule.agents[i].vector_id
            if vector_id % 10 == 3:
                I_v_history += self.schedule.agents[i].history
        self.I_v_history = I_v_history

        # N_h history
        N_h_history = np.zeros(self.tau * self.n_step_per_day)
        for i in range(self.n_agent_h):
            N_h_history += self.schedule.agents[i].history
        self.N_h_history = N_h_history

        # lambda_h history
        self.lambda_h_history = self.b_history * self.beta_hv * self.I_v_history / self.N_h_history

        # History for temperature-dependent parameters
        self.b_history0 = np.zeros(self.tau * self.n_step_per_day)
        self.mu_a_history0 = np.zeros(self.tau * self.n_step_per_day)
        self.delta_a_history0 = np.zeros(self.tau * self.n_step_per_day)
        self.mu_v_history0 = np.zeros(self.tau * self.n_step_per_day)
        self.delta_v_history0 = np.zeros(self.tau * self.n_step_per_day)

    def step(self):

        # Get current time index
        self.day_index = int(np.ceil(self.schedule.time / self.n_step_per_day) + self.day_init - 1)
        self.year_index = int(np.ceil(self.day_index / 365))

        # Assign current value of temperature dependent parameters
        self.b_ = self.b[self.day_index]
        self.mu_a_ = self.mu_a[self.day_index]
        self.delta_a_ = self.delta_a[self.day_index]
        self.mu_v_ = self.mu_v[self.day_index]
        self.delta_v_ = self.delta_v[self.day_index]

        self.b_history0 = np.append(self.b_history0, self.b_)
        self.mu_a_history0 = np.append(self.mu_a_history0, self.mu_a_)
        self.delta_a_history0 = np.append(self.delta_a_history0, self.delta_a_)
        self.mu_v_history0 = np.append(self.mu_v_history0, self.mu_v_)
        self.delta_v_history0 = np.append(self.delta_v_history0, self.delta_v_)

        # Update b
        self.b_history = np.append(self.b_history, self.b_)

        # Update I_v
        I_v_ = 0
        for i in self.unique_id_v:
            vector_id = self.schedule.agents[i].vector_id
            if vector_id % 10 == 3:
                I_v_ += self.schedule.agents[i].sol[-1]
        self.I_v_history = np.append(self.I_v_history, I_v_)

        # Update N_h
        N_h_ = 0
        for i in range(self.n_agent_h):
            N_h_ += self.schedule.agents[i].sol[-1]
        self.N_h_history = np.append(self.N_h_history, N_h_)

        # Update lambda_h
        lambda_h_ = self.b_history[-1] * self.beta_hv * self.I_v_history[-1] / self.N_h_history[-1]
        self.lambda_h_history = np.append(self.lambda_h_history, lambda_h_)

        # I_h
        I_h_ = 0
        for i in range(self.n_agent_h):
            human_id = self.schedule.agents[i].human_id
            if human_id % 10 == 2:
                I_h_ += self.schedule.agents[i].sol[-1]

        # lambda_v
        self.lambda_v_ = self.b_ * self.beta_vh * I_h_ / N_h_

        # Supply susceptible mosquitoes
        for i in self.unique_id_v:
            vector_id = self.schedule.agents[i].vector_id
            if vector_id % 10 == 1:
                S_v_ = self.schedule.agents[i].sol[-1]
                for day in self.tspan_vsupply:
                    if self.day_index + 1 == day:
                        S_v_ += self.N_h0 / 20 / self.n_region

                self.schedule.agents[i].S_v_supplied = S_v_

        # Step forward
        self.schedule.step()