# python sliar_rl.py n_episodes=10000 eps_start=0.0
# hydra list 호출
# python sliar_rl.py nu_actions="[0,1]" tau_actions="[0,1]"

import random
import hydra
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dqn_agent import Agent
from omegaconf import DictConfig, OmegaConf
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
from itertools import product

# nu: vaccine, tau: treatment, sigma: social distancing
@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf : DictConfig) -> None:
    # Control
    nu_actions = conf.nu_actions
    tau_actions = conf.tau_actions
    sigma_actions = conf.sigma_actions

    # Check the actions list
    ACTIONS = list(product(nu_actions, tau_actions, sigma_actions))
    print(ACTIONS)
    action_size = len(ACTIONS)
    print(f"actions size = {len(ACTIONS)}")

    def sliar(y, t, beta, sigma, kappa, alpha, tau, p, eta, epsilon, q, delta, nu):
        S, L, I , A = y
        dydt = np.array([- beta * (1-conf.sigma_max * sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - conf.nu_max * nu * S,
                        beta * (1-conf.sigma_max * sigma) * S * (epsilon * L + (1 - q) * I + delta * A) - kappa * L,
                        p * kappa * L - alpha * I - conf.tau_max * tau * I,
                        (1 - p) * kappa * L  - eta * A])
        return dydt

    class SliarEnvironment:
        def __init__(self, S0=1000000, L0=0, I0 = 1, A0 = 0):
            self.state = np.array([S0, L0, I0, A0])
            self.kappa = 0.526
            self.alpha = 0.244
            self.p = 0.667
            self.eta = 0.244
            self.epsilon = 0
            self.q = 0.5
            self.delta = 1
            self.R0 = 1.9847
            self.beta = self.R0/(S0 * ((self.epsilon / self.kappa) + ((1 - self.q)*self.p/self.alpha) + (self.delta*(1-self.p)/self.eta)))
            self.P = 1
            self.Q = conf.Q
            self.R = conf.R
            self.W = conf.W

        def reset(self, S0=1000000, L0=0, I0 = 1, A0 = 0):
            self.state = np.array([S0, L0, I0, A0])
            self.kappa = 0.526
            self.alpha = 0.244
            self.p = 0.667
            self.eta = 0.244
            self.epsilon = 0
            self.q = 0.5
            self.delta = 1
            self.R0 = 1.9847
            self.beta = self.R0/(S0 * ((self.epsilon / self.kappa) + ((1 - self.q)*self.p/self.alpha) + (self.delta*(1-self.p)/self.eta)))
            self.P = 1
            self.Q = conf.Q
            self.R = conf.R
            self.W = conf.W
            return self.state

        def step(self, action):
            nu, tau, sigma = ACTIONS[action]
            sol = odeint(sliar, self.state, np.linspace(0, 1, 101),
                        args=(self.beta, sigma, self.kappa, self.alpha, tau, self.p, self.eta, self.epsilon, self.q, self.delta, nu))
            new_state = sol[-1, :]
            S0, L0, I0, A0 = self.state
            S, L, I, A = new_state
            self.state = new_state
            # cost = PI + Qu^2 // P = 1, Q = 10e-06
            reward = - self.P * I - self.Q * ((conf.nu_max * nu) ** 2) - self.R * ((conf.tau_max * tau) ** 2) - self.W * ((conf.sigma_max * sigma) ** 2)
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
        actions.append(action)
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
    plt.title('SLIAR model without control')
    plt.xlabel('day')
    plt.savefig('SLIAR_wo_control.png', dpi=300)
    plt.show(block=False)


    # 2. Train DQN Agent
    env = SliarEnvironment()
    agent = Agent(state_size=4, action_size=action_size, seed=0)
    ## Parameters
    n_episodes=conf.n_episodes
    max_t=conf.tf_sliar
    eps_start=conf.eps_start
    eps_end=conf.eps_end
    eps_decay=conf.eps_decay

    ## Loop to learn
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    eps_window = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        states = state
        score = 0
        actions = []
        ACTIONSS = []
        for t in range(max_t):
            action = agent.act(state, eps)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            states = np.vstack((states, next_state))
            state = next_state
            score += reward
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        eps_window.append(eps)
        print('\rEpisode {}\tAverage Score: {:,.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 1000 == 0:
            print('\rEpisode {}\tAverage Score: {:,.2f}'.format(i_episode, np.mean(scores_window)))
            print(np.array(actions)[:5], eps)
            state = env.reset()
            max_t = 300
            states = state
            actions = []
            ACTIONSS = []
            score = 0
            for t in range(max_t):
                action = agent.act(state, eps=0.0)
                ACTION = ACTIONS[action]
                ACTIONSS = np.append(ACTIONSS, ACTION)
                actions = np.append(actions, action)
                next_state, reward, done, _ = env.step(action)
                states = np.vstack((states, next_state))
                state = next_state

            ACTIONSS = np.array(ACTIONSS).reshape(max_t, 3)
            S, L, I, A = np.hsplit(states, 4)
            nu_, tau_, sigma_ = np.hsplit(ACTIONSS, conf.action_dim)
            I_mid = (I[1:] + I[:-1]) / 2.
            nu_mid = (nu_[1:] + nu_[:-1]) / 2.
            tau_mid = (tau_[1:] + tau_[:-1]) / 2.
            sigma_mid = (sigma_[1:] + sigma_[:-1]) / 2.
            cost1 = np.sum(I_mid.flatten())
            cost2 = np.sum((conf.nu_max * nu_mid.flatten()) ** 2)
            cost3 = np.sum((conf.tau_max * tau_mid.flatten()) ** 2)
            cost4 = np.sum((conf.sigma_max * sigma_mid.flatten()) ** 2)
            cost = cost1 + conf.Q * cost2 + conf.R * cost3 + conf.W * cost4

            plt.clf()
            plt.plot(scores)
            plt.grid()
            plt.ylabel('cumulative future reward')
            plt.xlabel('episode')
            plt.savefig('SLIAR_score'+str(i_episode)+'.png', dpi=300)
            plt.show(block=False)

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
            plt.title('SLIAR model with control : n = '+str(i_episode))
            plt.xlabel('day')
            plt.savefig('SLIAR_w_control'+str(i_episode)+'.png', dpi=300)
            plt.clf()
            plt.plot(range(max_t), nu_, 'k+--', label = 'Vaccine')
            plt.plot(range(max_t), tau_, 'bx--', label = 'Treatment')
            plt.plot(range(max_t), sigma_, 'r1--', label = 'Social Distancing')
            plt.grid()
            plt.legend()
            plt.title('Control('+ str(cost)+') : n = '+str(i_episode))
            plt.xlabel('day')
            plt.savefig('SLIAR_control_u.png'+str(i_episode)+'.png', dpi=300)

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

    plt.clf()
    plt.plot(eps_window)
    plt.grid()
    plt.title('The change of epsilon'+str(eps_start))
    plt.ylabel('epsilon')
    plt.xlabel('episode')
    plt.savefig('epsilon.png', dpi=300)
    plt.show(block=False)

    # 3. Visualize Controlled SIR Dynamics
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    env = SliarEnvironment()
    state = env.reset()
    max_t = 300
    states = state
    actions = []
    ACTIONSS = []
    score = 0
    for t in range(max_t):
        action = agent.act(state, eps=0.0)
        ACTION = ACTIONS[action]
        ACTIONSS = np.append(ACTIONSS, ACTION)
        actions = np.append(actions, action)
        next_state, reward, done, _ = env.step(action)
        states = np.vstack((states, next_state))
        state = next_state

    ACTIONSS = np.array(ACTIONSS).reshape(max_t, 3)
    S, L, I, A = np.hsplit(states, 4)
    nu_, tau_, sigma_ = np.hsplit(ACTIONSS, conf.action_dim)
    I_mid = (I[1:] + I[:-1]) / 2.
    nu_mid = (nu_[1:] + nu_[:-1]) / 2.
    tau_mid = (tau_[1:] + tau_[:-1]) / 2.
    sigma_mid = (sigma_[1:] + sigma_[:-1]) / 2.
    cost1 = np.sum(I_mid.flatten())
    cost2 = np.sum((conf.nu_max * nu_mid.flatten()) ** 2)
    cost3 = np.sum((conf.tau_max * tau_mid.flatten()) ** 2)
    cost4 = np.sum((conf.sigma_max * sigma_mid.flatten()) ** 2)
    cost = cost1 + conf.Q * cost2 + conf.R * cost3 + conf.W * cost4

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
    plt.plot(range(max_t), nu_, '.-k', label = 'Vaccine')
    plt.plot(range(max_t), tau_, '.-b', label = 'Treatment')
    plt.plot(range(max_t), sigma_, '.-r', label = 'Social Distancing')
    plt.grid()
    plt.legend()
    plt.title('Control('+ str(cost)+')')
    plt.xlabel('day')
    plt.savefig('SLIAR_control_u.png', dpi=300)
    plt.show(block=False)


if __name__ == '__main__':
    main()