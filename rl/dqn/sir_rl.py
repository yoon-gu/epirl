import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from dqn_agent import Agent

def sir(y, t, beta, gamma, u):
    S, I = y
    dydt = np.array([-beta * S * I - u * S, beta * S * I - gamma * I])
    return dydt

class SirEnvironment:
    def __init__(self, S0=990, I0=10):
        self.state = np.array([S0, I0])
        self.beta = 0.002
        self.gamma = 0.5

    def reset(self, S0=990, I0=10):
        self.state = np.array([S0, I0])
        self.beta = 0.002
        self.gamma = 0.5
        return self.state

    def step(self, action):
        sol = odeint(sir, self.state, np.linspace(0, 1, 101), args=(self.beta, self.gamma, action))
        new_state = sol[-1, :]
        S0, I0 = self.state
        S, I = new_state
        self.state = new_state
        reward = - I
        done = True if new_state[1] < 1.0 else False
        return (new_state, reward, done, 0)


plt.rcParams['figure.figsize'] = (8, 4.5)

# 1. Without Control
env = SirEnvironment()
state = env.reset()
max_t = 30
states = state
actions = []
for t in range(max_t):
    action = 0
    next_state, reward, done, _ = env.step(action)
    states = np.vstack((states, next_state))
    state = next_state

plt.clf()
plt.plot(range(max_t+1), states[:,0].flatten(), '.-')
plt.plot(range(max_t+1), states[:,1].flatten(), '.-')
plt.grid()
plt.title('SIR model without control')
plt.xlabel('day')
plt.savefig('SIR_wo_control.png', dpi=300)
plt.show(block=False)

# 2. Train DQN Agent
env = SirEnvironment()
agent = Agent(state_size=2, action_size=2, seed=0)
## Parameters
n_episodes=2000
max_t=30
eps_start=1.0 # Too large epsilon for a stable learning
eps_end=0.001
eps_decay=0.995

## Loop to learn
scores = []                        # list containing scores from each episode
scores_window = deque(maxlen=100)  # last 100 scores
eps = eps_start                    # initialize epsilon
for i_episode in range(1, n_episodes+1):
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
    print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
    if i_episode % 400 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
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
plt.savefig('SIR_score.png', dpi=300)
plt.show(block=False)

# 3. Visualize Controlled SIR Dynamics
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
env = SirEnvironment()
state = env.reset()
max_t = 30
states = state
actions = []
for t in range(max_t):
    action = agent.act(state, eps=0.0)
    actions = np.append(actions, action)
    next_state, reward, done, _ = env.step(action)
    states = np.vstack((states, next_state))
    state = next_state

plt.clf()
plt.plot(range(max_t+1), states[:,0].flatten(), '.-')
plt.plot(range(max_t+1), states[:,1].flatten(), '.-')
plt.grid()
plt.title('SIR model with control')
plt.xlabel('day')
plt.savefig('SIR_w_control.png', dpi=300)
plt.show(block=False)

plt.clf()
plt.plot(range(max_t), actions, '.-k')
plt.grid()
plt.title('Vaccine Control')
plt.xlabel('day')
plt.savefig('SIR_control_u.png', dpi=300)
plt.show(block=False)