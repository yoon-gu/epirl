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
import os

npath = os.getcwd()

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

        def reset(self, S0=conf.S0, I0=conf.I0):
            self.state = np.array([S0, I0])
            self.beta = conf.beta
            self.gamma = conf.gamma
            return self.state

        def step(self, action):
            sol = odeint(sir, self.state, np.linspace(0, 1, 101), args=(self.beta, self.gamma, self.nu*action))
            # 1%만 백신접종
            
            new_state = sol[-1, :]
            S0, I0 = self.state
            S, I = new_state
            self.state = new_state
            reward = - I - action
            done = True if new_state[1] < 1.0 else False
            return (new_state, reward, done, 0)
        
        
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

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
    fig.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/SIR_wo_vac.png")
    
    env = SirEnvironment()
    state = env.reset()
    agent = Agent(state_size=2, action_size=2, seed=0)
    
    # triangle plot
    S = np.linspace(0, 10000, 101)
    I = np.linspace(0, 10000, 101)

    SS, II = np.meshgrid(S, I)

    vf = np.zeros((len(I), len(S)))
    af = np.zeros((len(I), len(S)))

    for si, s in enumerate(S):
        for ii, i in enumerate(I):
            v = agent.qnetwork_local.forward(torch.tensor([float(s), float(i)]).to(device))
            v = v.detach().cpu().numpy()
            vf[si, ii] = np.max(v)
            af[si, ii] = np.argmax(v)

    vf[SS + II > 10000] = None
    af[SS + II > 10000] = None
    
    fig = go.Figure(data =
    [go.Contour(
        z=-vf,
        x=S,
        y=I),
    go.Scatter(mode='markers',
               x=states[:,0],
               y=states[:,1],
               marker=dict(color='rgba(230, 240, 255, 0.5)',
                           size=8))])
    fig.update_layout(title_text='Value')
    fig.update_xaxes(title_text='S')
    fig.update_yaxes(title_text='I')
    wandb.log({"Value (initial)": fig})
    fig.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/value_initial.png")
    
    fig = go.Figure(data =
    go.Contour(
        z=af,
        x=S,
        y=I
    ))
    fig.update_layout(title_text='Action')
    fig.update_xaxes(title_text='S')
    fig.update_yaxes(title_text='I')
    wandb.log({"Action (initial)": fig})
    fig.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/action_initial.png")
    

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
    fig.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/SIR_w_fvac.png")


    # 2. Train DQN Agent
    env = SirEnvironment()
    agent = Agent(state_size=states.shape[1], action_size=2, seed=0)
    ## Parameters
    n_episodes=conf.n_episodes
    repeat_times=conf.repeat_times
    max_t=conf.tf
    eps_start=conf.eps_start
    eps_end= min(eps_start, conf.eps_end)
    eps_decay=conf.eps_decay
    
    
    ## Loop to learn
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon

    for j in range(1, repeat_times+1):
        
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
            
            if i_episode % 100 == 0:
                print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
                print(np.array(actions)[:5], eps)
            if np.mean(scores_window)>=200.0:
                print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode-100, np.mean(scores_window)))
                torch.save(agent.qnetwork_local.state_dict(), 'checkpoint2.pth')
                break

        torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
        
        fig2 = go.Figure(data = 
                         [go.Scatter(x=list(range(1, n_episodes*j+1)),
                                 y=scores)]
        )
        fig2.update_layout(title_text='Score')
        fig2.update_xaxes(title_text='number of episodes')
        fig2.update_yaxes(title_text='score')
        fig2.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/Score_{j*n_episodes}.png")

        # 3. Visualize Controlled SIR Dynamics
        agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))
        env2 = SirEnvironment()
        state2 = env2.reset()
        max_t = conf.tf
        states2 = state2
        reward_sum2 = 0.
        actions2 = []
        for t in range(max_t):
            action2 = agent.act(state2, eps=0.0)
            run.log({'Vaccine': action2, 't':t})
            actions2 = np.append(actions2, action2)
            next_state2, reward2, done, _ = env2.step(action2)
            reward_sum2 += reward2
            states2 = np.vstack((states2, next_state2))
            state2 = next_state2
            
        # Create figure with secondary y-axis
        fig = make_subplots(specs=[[{"secondary_y": True}]])
    
        # Add traces
        fig.add_trace(
            go.Scatter(x=list(range(max_t+1)), y=states2[:,0].flatten(), name="susceptible",
                mode='lines+markers'),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=list(range(max_t+1)), y=states2[:,1].flatten(), name="infected",
                mode='lines+markers'),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=list(range(max_t+1)), y=actions2, name="vaccine",
                mode='lines+markers'),
            secondary_y=True,
        )
        # Add figure title
        fig.update_layout(
            title_text=f'{reward_sum2:.2f}: SIR model with control'
        )
        # Set x-axis title
        fig.update_xaxes(title_text="day")
        # Set y-axes titles
        fig.update_yaxes(title_text="Population", secondary_y=False)
        fig.update_yaxes(title_text="Vaccine", secondary_y=True)

        wandb.log({f"SIR with vaccine{j*n_episodes}": fig})
        fig.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/SIR_w_vac_{j*n_episodes}.png")
        
        S = np.linspace(0, 10000, 101)
        I = np.linspace(0, 10000, 101)

        SS, II = np.meshgrid(S, I)

        vf = np.zeros((len(I), len(S)))
        af = np.zeros((len(I), len(S)))

        for si, s in enumerate(S):
            for ii, i in enumerate(I):
                v = agent.qnetwork_local.forward(torch.tensor([float(s), float(i)]).to(device))
                v = v.detach().cpu().numpy()
                vf[si, ii] = np.max(v)
                af[si, ii] = np.argmax(v)

        vf[SS + II > 10000] = None
        af[SS + II > 10000] = None
        
        fig3 = go.Figure(data =
                        [go.Contour(
                            z=-vf,
                            x=S,
                            y=I
                            ),
                         go.Scatter(
                            mode='markers',
                            x=states2[:,0],
                            y=states2[:,1],
                            marker=dict(color='rgba(230, 240, 255, 0.5)',
                                        size=8))]
                        )
        fig3.update_layout(title_text='Value')
        fig3.update_xaxes(title_text='S')
        fig3.update_yaxes(title_text='I')
        wandb.log({f"Value {n_episodes*j}": fig3})
        fig3.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/value_{n_episodes*j}.png")
        
        
        fig3 = go.Figure(data =
                        go.Contour(
                            z=af,
                            x=S,
                            y=I
                            ))
        fig3.update_layout(title_text='Action')
        fig3.update_xaxes(title_text='S')
        fig3.update_yaxes(title_text='I')
        wandb.log({f"Action {n_episodes*j}": fig3})
        fig3.write_image(f"{npath}/images/nu{int(conf.nu*100)}beta{int(conf.beta*10000)}/action_{n_episodes*j}.png")
        
        
    run.summary.update(conf)
    run.summary['Final_Reward'] = reward_sum
    wandb.finish()

if __name__ == '__main__':
    main()
