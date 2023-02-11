import os
import hydra
from hydra.utils import instantiate
import numpy as np
import plotly.graph_objects as go

from plotly.subplots import make_subplots
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import PPO, DQN, A2C, SAC, DDPG
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    EveryNTimesteps,
    StopTrainingOnMaxEpisodes,
    StopTrainingOnNoModelImprovement,
    StopTrainingOnRewardThreshold,
)

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(conf: DictConfig):
    train_env = instantiate(conf.sir)
    check_env(train_env)
    log_dir = "./sir_ppo_log"
    os.makedirs(log_dir, exist_ok=True)
    train_env = Monitor(train_env, log_dir)
    policy_kwargs = dict(
                            # activation_fn=torch.nn.ReLU,
                            # net_arch=[32, 32]
                        )
    model = PPO("MlpPolicy", train_env, verbose=0,
                policy_kwargs=policy_kwargs)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
    print("Before:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    eval_env = instantiate(conf.sir)
    eval_callback = EvalCallback(
            eval_env,
            eval_freq=5000,
            warn=False,
            log_path='eval_log',
            best_model_save_path='best_model'
        )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/',
                                             name_prefix='rl_model')
    callback = CallbackList([checkpoint_callback, eval_callback])

    model.learn(total_timesteps=conf.n_steps, callback=callback)
    model.save("sir_ppo")
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
    print("After:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    # Visualize Controlled SIR Dynamics
    state = eval_env.reset()
    max_t = conf.sir.tf
    states = state
    reward_sum = 0.
    actions = []
    for t in range(max_t):
        action, _states = model.predict(state)
        actions = np.append(actions, conf.sir.v_min + conf.sir.v_max * (1.0 + action[0]) / 2.0)
        next_state, reward, _, _ = eval_env.step(action)
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