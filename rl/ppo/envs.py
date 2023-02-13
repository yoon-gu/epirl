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
    def __init__(self, S0, I0, beta, gamma, v_min, v_max, tf):
        self.state = np.array([S0, I0])
        self.beta = beta
        self.gamma = gamma
        self.v_min = v_min
        self.v_max = v_max
        self.S0 = S0
        self.I0 = I0
        self.tf = tf
        self.observation_space = gym.spaces.Box(
                    low=np.array([0.0, 0.0], dtype=np.float32),
                    high=np.array([1000.0, 1000.0], dtype=np.float32),
                    dtype=np.float32)
        self.action_space = gym.spaces.Box(
                    low=np.array([-1.0], dtype=np.float32),
                    high=np.array([1.0], dtype=np.float32),
                    dtype=np.float32)
        self.time = 0.0
        self.dt = 1.0

    def reset(self):
        self.days = [self.time]
        self.susceptible = [self.S0]
        self.infected = [self.I0]
        self.actions = []
        self.time = 0.0
        self.state = np.array([self.S0, self.I0])
        return np.array(self.state, dtype=np.float32)

    def step(self, action):
        vaccine = self.v_min + self.v_max * (action[0] + 1.0) / 2.0
        self.actions.append(vaccine)

        sol = odeint(sir, self.state, np.linspace(0, 1, 101), args=(self.beta, self.gamma, vaccine))
        self.time += self.dt
        new_state = sol[-1, :]
        S0, I0 = self.state
        S, I = new_state
        self.state = new_state
        reward = - I - 10*vaccine

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