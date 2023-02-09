import gym
import os
import hydra
import torch
import numpy as np
import plotly.graph_objects as go
from scipy.integrate import odeint
from plotly.subplots import make_subplots
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):

    beta= conf.beta
    gamma= conf.gamma
    tf= conf.tf
    S0= conf.S0
    I0= conf.I0
    v_min = conf.v_min
    v_max = conf.v_max

    def sir(y, t, beta, gamma, u):
        S, I = y
        dydt = np.array([-beta * S * I - u * S, beta * S * I - gamma * I])
        return dydt

    class SirEnvironment(gym.Env):
        def __init__(self, S0=S0, I0=I0, beta=beta, gamma=gamma, v_min=v_min, v_max=v_max):
            self.state = np.array([S0, I0])
            self.beta = beta
            self.gamma = gamma
            self.v_min = v_min
            self.v_max = v_max
            self.observation_space = gym.spaces.Box(low=np.array([0.0, 0.0]), high=np.array([1000.0, 1000.0]), dtype=np.float32)
            self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        def reset(self, S0=S0, I0=I0):
            self.state = np.array([S0, I0])
            self.beta = beta
            self.gamma = gamma
            return np.array(self.state, dtype=np.float32)

        def step(self, action):
            vaccine = self.v_min + self.v_max * (action[0] + 1.0) / 2.0
            sol = odeint(sir, self.state, np.linspace(0, 1, 101), args=(self.beta, self.gamma, vaccine))
            new_state = sol[-1, :]
            S0, I0 = self.state
            S, I = new_state
            self.state = new_state
            reward = - I - 10*vaccine
            done = True if new_state[1] < 1.0 else False
            return (np.array(new_state, dtype=np.float32), reward, done, {})


    env = SirEnvironment()
    check_env(env)
    log_dir = "./sir_ppo_log"
    os.makedirs(log_dir, exist_ok=True)
    env = Monitor(env, log_dir)
    policy_kwargs = dict(
                            # activation_fn=torch.nn.ReLU,
                            # net_arch=[32, 32]
                        )
    model = PPO("MlpPolicy", env, verbose=0, policy_kwargs=policy_kwargs)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print("Before")
    print(f"mean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    model.learn(total_timesteps=conf.n_steps)
    model.save("sir_ppo")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print("After")
    print(f"mean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    # Visualize Controlled SIR Dynamics
    env = SirEnvironment()
    state = env.reset()
    max_t = tf
    states = state
    reward_sum = 0.
    actions = []
    for t in range(max_t):
        action, _states = model.predict(state)
        actions = np.append(actions, v_min + v_max * (1.0 + action[0]) / 2.0)
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
    fig.show()

if __name__ == '__main__':
    main()