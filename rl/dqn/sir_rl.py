import random
import wandb
import hydra
import torch
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from collections import deque
from scipy.integrate import odeint
from dqn_agent import Agent
from omegaconf import DictConfig, OmegaConf

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig) -> None:
    run = wandb.init(project='SIR+DQN', job_type='Train an agent')

    def sir(y, t, beta, gamma, u):
        S, I = y
        dydt = np.array([-beta * S * I - u * S, beta * S * I - gamma * I])
        return dydt

    class SirEnvironment:
        def __init__(self, S0=conf.S0, I0=conf.I0):
            self.state = np.array([S0, I0])
            self.beta = conf.beta
            self.gamma = conf.gamma
            self.nu = conf.nu
            self.w_action = conf.w_action

        def reset(self, S0=conf.S0, I0=conf.I0):
            self.state = np.array([S0, I0])
            self.beta = conf.beta
            self.gamma = conf.gamma
            return self.state

        def step(self, action):
            sol = odeint(sir, self.state, np.linspace(0, 1, 101), args=(self.beta, self.gamma, self.nu*action))
            new_state = sol[-1, :]
            S0, I0 = self.state
            S, I = new_state
            self.state = new_state
            reward = - I - action*self.w_action
            done = True if new_state[1] < 1.0 else False
            return (new_state, reward, done, 0)

    # 1-1. Without Control
    env = SirEnvironment()
    state = env.reset()
    max_t = conf.tf
    states = state
    reward_sum = 0.0
    actions = []
    for t in range(max_t):
        action = 0
        next_state, reward, done, _ = env.step(action)
        states = np.vstack((states, next_state))
        reward_sum += reward
        actions.append(action)
        state = next_state

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=states[:,0].flatten(), name="susceptible",
            mode='lines+markers'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=states[:,1].flatten(), name="infected",
            mode='lines+markers'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=actions, name="vaccine",
            mode='lines+markers'),
        secondary_y=True,
    )
    # Add figure title
    fig.update_layout(
        title_text=f'{reward_sum:.2f}: SIR model without control'
    )
    # Set x-axis title
    fig.update_xaxes(title_text="day")
    # Set y-axes titles
    fig.update_yaxes(title_text="Population", secondary_y=False)
    fig.update_yaxes(title_text="Vaccine", secondary_y=True)
    wandb.log({"SIR without vaccine": fig})

    # 1-2. With Full Control
    env = SirEnvironment()
    state = env.reset()
    max_t = conf.tf
    states = state
    actions = []
    reward_sum = 0.
    for t in range(max_t):
        action = 1
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        actions.append(action)
        states = np.vstack((states, next_state))
        state = next_state

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=states[:,0].flatten(), name="susceptible",
            mode='lines+markers'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=states[:,1].flatten(), name="infected",
            mode='lines+markers'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=actions, name="vaccine",
            mode='lines+markers'),
        secondary_y=True,
    )
    # Add figure title
    fig.update_layout(
        title_text=f'{reward_sum:.2f}: SIR model with full control'
    )
    # Set x-axis title
    fig.update_xaxes(title_text="day")
    # Set y-axes titles
    fig.update_yaxes(title_text="Population", secondary_y=False)
    fig.update_yaxes(title_text="Vaccine", secondary_y=True)
    wandb.log({"SIR with full vaccine": fig})


    # 2. Train DQN Agent
    env = SirEnvironment()
    agent = Agent(state_size=states.shape[1], action_size=2, seed=0)
    ## Parameters
    n_episodes=conf.n_episodes
    max_t=conf.tf
    eps_start=conf.eps_start
    eps_end= min(eps_start, conf.eps_end)
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
            action = agent.act(state, eps)
            actions.append(action)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break
        run.log({'Reward': score, 'eps': eps, 'episode': i_episode})
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

    # 3. Visualize Controlled SIR Dynamics
    agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
    env = SirEnvironment()
    state = env.reset()
    max_t = conf.tf
    states = state
    reward_sum = 0.
    actions = []
    for t in range(max_t):
        action = agent.act(state, eps=0.0)
        run.log({'Vaccine': action, 't':t})
        actions = np.append(actions, action)
        next_state, reward, done, _ = env.step(action)
        reward_sum += reward
        states = np.vstack((states, next_state))
        state = next_state

    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    # Add traces
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=states[:,0].flatten(), name="susceptible",
            mode='lines+markers'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=states[:,1].flatten(), name="infected",
            mode='lines+markers'),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(x=list(range(max_t+1)), y=actions, name="vaccine",
            mode='lines+markers'),
        secondary_y=True,
    )
    # Add figure title
    fig.update_layout(
        title_text=f'{reward_sum:.2f}: SIR model with control'
    )
    # Set x-axis title
    fig.update_xaxes(title_text="day")
    # Set y-axes titles
    fig.update_yaxes(title_text="Population", secondary_y=False)
    fig.update_yaxes(title_text="Vaccine", secondary_y=True)

    wandb.log({"SIR with vaccine": fig})
    run.summary.update(conf)
    run.summary['Final_Reward'] = reward_sum
    wandb.finish()

if __name__ == '__main__':
    main()
