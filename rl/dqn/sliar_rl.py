import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dqn_agent import Agent
from tqdm import tqdm

def sliar(y, t, beta, sigma, kappa, alpha, tau, p, eta, epsilon, q, delta, u):
    S, L, I , A = y
    dydt = np.array([- beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - u * S,
                    beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - kappa * L,
                    p * kappa * L - alpha * I - tau * I,
                    (1 - p) * kappa * L  - eta * A])
    return dydt

class SliarEnvironment:
    def __init__(self, S0=1000000, L0=0, I0 = 1, A0 = 0):
        self.state = np.array([S0, L0, I0, A0])
        self.beta = 0.000000527
        self.sigma = 0
        self.kappa = 0.526
        self.alpha = 0.244
        self.tau = 0
        self.p = 0.667
        self.eta = 0.244
        self.epsilon = 0
        self.q = 0.5
        self.delta = 1


    def reset(self, S0=1000000, L0=0, I0 = 1, A0 = 0):
        self.state = np.array([S0, L0, I0, A0])
        self.beta = 0.000000527
        self.sigma = 0
        self.kappa = 0.526
        self.alpha = 0.244
        self.tau = 0
        self.p = 0.667
        self.eta = 0.244
        self.epsilon = 0
        self.q = 0.5
        self.delta = 1
        return self.state

    def step(self, action):
        sol = odeint(sliar, self.state, np.linspace(0, 1, 101),
                    args=(self.beta, self.sigma, self.kappa, self.alpha, self.tau, self.p, self.eta, self.epsilon, self.q, self.delta, action))
        new_state = sol[-1, :]
        S0, L0, I0, A0 = self.state
        S, L, I, A = new_state
        self.state = new_state
        # cost = PI + Qu^2 // P = 1, Q = 10e-06
        reward = - I - action ** 2
        done = True if new_state[2] < 1.0 else False
        return (new_state, reward, False, 0)


for eps_ in np.linspace(0., 1, 11):
    for n_episodes in [10, 100, 1000, 2000, 5000]:
        filename1 = f"images/si_eps_{eps_:.2f}_n_{n_episodes:05d}.png"
        filename2 = f"images/score_eps_{eps_:.2f}_n_{n_episodes:05d}.png"
        filename3 = f"images/vaccine_eps_{eps_:.2f}_n_{n_episodes:05d}.png"

        plt.rcParams['figure.figsize'] = (8, 4.5)

        # 1. Without Control
        env = SliarEnvironment()
        state = env.reset()
        max_t = 300
        states = state
        actions = []
        for t in range(max_t):
            action = 0
            next_state, reward, done, _ = env.step(action)
            states = np.vstack((states, next_state))
            state = next_state

        # 2. Train DQN Agent
        env = SliarEnvironment()
        agent = Agent(state_size=4, action_size=2, seed=0)
        ## Parameters
        max_t=300
        eps_start=eps_ # Too large epsilon for a stable learning
        eps_end=0.000
        eps_decay=0.995

        ## Loop to learn
        scores = []                        # list containing scores from each episode
        scores_window = deque(maxlen=100)  # last 100 scores
        eps = eps_start                    # initialize epsilon
        for i_episode in tqdm(range(1, n_episodes+1), desc=f"eps={eps_:.2f}+n={n_episodes:05}"):
            state = env.reset()
            score = 0
            actions = []
            for t in range(max_t):
                action = agent.act(state, eps)
                actions.append(action)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)
                state = next_state
                score += reward
                if done:
                    break
            scores_window.append(score)       # save most recent score
            scores.append(score)              # save most recent score
            eps = max(eps_end, eps_decay*eps) # decrease epsilon

        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

        plt.clf()
        plt.plot(scores)
        plt.grid()
        plt.ylabel('cumulative future reward')
        plt.xlabel('episode')
        plt.savefig(filename2, dpi=300)
        plt.show(block=False)

        # 3. Visualize Controlled SIR Dynamics
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
        env = SliarEnvironment()
        state = env.reset()
        max_t = 300
        states = state
        actions = []
        for t in range(max_t):
            action = agent.act(state, eps=0.0)
            actions = np.append(actions, action)
            next_state, reward, done, _ = env.step(action)
            states = np.vstack((states, next_state))
            state = next_state


        plt.clf()
        plt.plot(range(max_t+1), states[:,1].flatten(), '.-', label = 'L')
        plt.plot(range(max_t+1), states[:,2].flatten(), '.-', label = 'I')
        plt.plot(range(max_t+1), states[:,3].flatten(), '.-', label = 'A')
        plt.grid()
        plt.legend()
        plt.title('SLIAR model with control')
        plt.xlabel('day')
        plt.savefig(filename1, dpi=300)
        plt.show(block=False)

        plt.clf()
        plt.plot(range(max_t), actions, '-k')
        plt.grid()
        plt.title('Vaccine Control')
        plt.xlabel('day')
        plt.ylim([-0.1, 1.1])
        plt.savefig(filename3, dpi=300)
        plt.show(block=False)

        plt.close('all')