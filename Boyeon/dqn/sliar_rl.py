# python sliar_rl.py n_episodes=10000 eps_start=0.0

import random
import hydra
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dqn_agent import Agent
from omegaconf import DictConfig, OmegaConf

ACTIONS = [(0, 0),
           (1, 0),
           (0, 1),
           (1, 1) ]

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    def sliar(y, t, beta, sigma, kappa, alpha, tau, p, eta, epsilon, q, delta, nu):
        S, L, I , A = y
        dydt = np.array([- beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - conf.nu_max * nu * S,
                        beta * (1-sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - kappa * L,
                        p * kappa * L - alpha * I - conf.tau_max * tau * I,
                        (1 - p) * kappa * L  - eta * A])
        return dydt

    class SliarEnvironment:
        def __init__(self, S0=1000000, L0=0, I0 = 1, A0 = 0):
            self.state = np.array([S0, L0, I0, A0])
            self.sigma = 0
            self.kappa = 0.526
            self.alpha = 0.244
            #self.tau = 0
            self.p = 0.667
            self.eta = 0.244
            self.epsilon = 0
            self.q = 0.5
            self.delta = 1
            self.R0 = 1.9847
            self.beta = self.R0/(S0 * ((self.epsilon / self.kappa) + ((1 - self.q)*self.p/self.alpha) + (self.delta*(1-self.p)/self.eta)))
            self.P = 1
            self.Q = 1E6
            self.R = 1E6
            self.W = 0


        def reset(self, S0=1000000, L0=0, I0 = 1, A0 = 0):
            self.state = np.array([S0, L0, I0, A0])
            self.sigma = 0
            self.kappa = 0.526
            self.alpha = 0.244
            #self.tau = 0
            self.p = 0.667
            self.eta = 0.244
            self.epsilon = 0
            self.q = 0.5
            self.delta = 1
            self.R0 = 1.9847
            self.beta = self.R0/(S0 * ((self.epsilon / self.kappa) + ((1 - self.q)*self.p/self.alpha) + (self.delta*(1-self.p)/self.eta)))
            self.P = 1
            self.Q = 1E6
            self.R = 1E6
            self.W = 0
            return self.state

        def step(self, action):
            nu, tau = ACTIONS[action]
            sol = odeint(sliar, self.state, np.linspace(0, 1, 101),
                        args=(self.beta, self.sigma, self.kappa, self.alpha, tau, self.p, self.eta, self.epsilon, self.q, self.delta, nu))
            new_state = sol[-1, :]
            S0, L0, I0, A0 = self.state
            S, L, I, A = new_state
            self.state = new_state
            # cost = PI + Qu^2 // P = 1, Q = 10e-06
            reward = - self.P * I - self.Q * ((conf.nu_max * nu) ** 2) - self.R * ((conf.tau_max * tau) ** 2)
            done = True if new_state[2] < 1.0 else False
            return (new_state, reward, False, 0)


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

    plt.clf()
    fig, ax1 = plt.subplots()
    ax1.plot(range(max_t+1), states[:,0], '.-b', label = 'S')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(range(max_t+1), states[:,1], '.-y', label = 'L')
    ax2.plot(range(max_t+1), states[:,2], '.-r', label = 'I')
    ax2.plot(range(max_t+1), states[:,3], '.-g', label = 'A')
    ax2.legend(loc='lower right')
    plt.grid()
    plt.legend()
    plt.title('SLIAR model w/o control')
    plt.xlabel('day')
    plt.savefig('SLIAR_wo_control.png', dpi=300)
    plt.show(block=False)

    # 2. Train DQN Agent
    env = SliarEnvironment()
    agent = Agent(state_size=4, action_size=4, seed=0)
    ## Parameters
    n_episodes=conf.n_episodes
    max_t=300
    eps_start=conf.eps_start
    eps_end=conf.eps_end
    eps_decay=conf.eps_decay

    ## Loop to learn
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        actions = []
        for t in range(max_t):
            action = agent.act(state,eps)
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
        print('\rEpisode {}\tAverage Score: {:,.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 400 == 0:
            print('\rEpisode {}\tAverage Score: {:,.2f}'.format(i_episode, np.mean(scores_window)))
            print(np.array(actions)[:5], eps)
        if np.mean(scores_window)>=200.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint2.pth')
            break

    torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')

    plt.clf()
    plt.plot(scores)
    plt.grid()
    plt.ylabel('cumulative future reward')
    plt.xlabel('episode')
    plt.savefig('SLIAR_score.png', dpi=300)
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
    fig, ax1 = plt.subplots()
    ax1.plot(range(max_t+1), states[:,0], '.-b', label = 'S')
    ax1.legend(loc='upper left')
    ax2 = ax1.twinx()
    ax2.plot(range(max_t+1), states[:,1], '.-y', label = 'L')
    ax2.plot(range(max_t+1), states[:,2], '.-r', label = 'I')
    ax2.plot(range(max_t+1), states[:,3], '.-g', label = 'A')
    ax2.legend(loc='lower right')
    plt.grid()
    plt.legend()
    plt.title('SLIAR model with control')
    plt.xlabel('day')
    plt.savefig('SLIAR_w_control.png', dpi=300)
    plt.show(block=False)

    plt.clf()
    plt.plot(range(max_t), actions, '.-k')
    plt.yticks([0,1,2,3], labels=['(0,0)', '(0,1)', '(1,0)', '(1,1)'])
    plt.grid()
    plt.title('Vaccine Control')
    plt.xlabel('day')
    plt.savefig('SLIAR_control_u.png', dpi=300)
    plt.show(block=False)


if __name__ == '__main__':
    main()