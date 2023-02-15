import gym
import numpy as np
import pandas as pd
from scipy.integrate import odeint

def sir(y, t, beta, gamma, u):
    S, I = y
    dydt = np.array([-beta * S * I - u * S, beta * S * I - gamma * I])
    return dydt

class SirEnvironment(gym.Env):
    beta: float
    gamma: float
    v_min: float
    v_max: float
    S0: float
    I0: float
    tf: float
    days: list
    susceptible: list
    infected: list
    actions: list
    vaccine_importance: float
    continuous_actions: bool
    def __init__(self, S0, I0, beta, gamma, v_min, v_max, tf, vaccine_importance, continuous_actions):
        self.state = np.array([S0, I0])
        self.beta = beta
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.S0 = S0
        self.I0 = I0
        self.tf = tf
        self.vaccine_importance = vaccine_importance
        self.continuous_actions = continuous_actions
        self.observation_space = gym.spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([1000.0, 1000.0], dtype=np.float32),
                    dtype=np.float32)
        if self.continuous_actions:
            self.action_space = gym.spaces.Box(
                        low=np.array([-1.0], dtype=np.float32),
                        high=np.array([1.0], dtype=np.float32),
                        dtype=np.float32)
        else:
            self.action_space = gym.spaces.Discrete(2)
        self.time = 0.0
        self.dt = 1.0

    def reset(self):
        self.time = 0.0
        self.days = [self.time]
        self.susceptible = [self.S0]
        self.infected = [self.I0]
        self.actions = []
        self.state = np.array([self.S0, self.I0])
        return np.array(self.state, dtype=np.float32)

    def action2vaccine(self, action):
        return self.v_min + self.v_max * (action[0] + 1.0) / 2.0

    def step(self, action):
        if self.continuous_actions:
            vaccine = self.action2vaccine(action)
        else:
            vaccine = self.v_min if action == 0 else self.v_max
        self.actions.append(vaccine)

        sol = odeint(sir, self.state, np.linspace(0, 1, 101), args=(self.beta, self.gamma, vaccine))
        self.time += self.dt
        new_state = sol[-1, :]
        S0, I0 = self.state
        S, I = new_state
        self.state = new_state
        reward = - I - self.vaccine_importance * vaccine

        self.days.append(self.time)
        self.susceptible.append(S)
        self.infected.append(I)

        done = True if self.time >= self.tf else False
        return (np.array(new_state, dtype=np.float32), reward, done, {})

    @property
    def dynamics(self):
        df= pd.DataFrame(dict(
                                days=self.days, 
                                susceptible=self.susceptible,
                                infected=self.infected,
                                vaccines=self.actions + [None]
                            )
                        )
        return df
