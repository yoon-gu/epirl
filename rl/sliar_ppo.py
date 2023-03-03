import os
import torch
import hydra
from hydra.utils import instantiate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from plotly.subplots import make_subplots
from omegaconf import DictConfig, OmegaConf
import stable_baselines3 as sb3
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
    ProgressBarCallback
)
# pip install git+https://github.com/carlosluis/stable-baselines3@cff332c29096e0095ceef20df70be66b1b82d44c

sns.set_theme(style="whitegrid")

@hydra.main(version_base=None, config_path="conf", config_name="ppo_sliar")
def main(conf: DictConfig):
    train_env = instantiate(conf.sliar)
    check_env(train_env)
    log_dir = "./sliar_ppo_log"
    os.makedirs(log_dir, exist_ok=True)
    train_env = Monitor(train_env, log_dir)
    policy_kwargs = dict(
                            # activation_fn=torch.nn.ReLU,
                            # net_arch=[256, 128, 64]
                        )
    Algorithm = getattr(sb3, conf.algorithm)
    model = Algorithm("MlpPolicy", train_env,
                policy_kwargs=policy_kwargs)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
    print("Before:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    eval_env = instantiate(conf.sliar)
    eval_callback = EvalCallback(
            eval_env,
            eval_freq=1000,
            verbose=0,
            warn=False,
            log_path='eval_log',
            best_model_save_path='best_model'
        )
    checkpoint_callback = CheckpointCallback(save_freq=1000, save_path='./checkpoints/',
                                             name_prefix=f"{conf.algorithm}")
    callback = CallbackList([checkpoint_callback, eval_callback, ProgressBarCallback()])

    model.learn(total_timesteps=conf.n_steps, callback=callback)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10)
    print("After:")
    print(f"\tmean_reward:{mean_reward:,.2f} +/- {std_reward:.2f}")

    os.makedirs('figures', exist_ok=True)
    df = pd.read_csv(f"{log_dir}/monitor.csv", skiprows=1)
    sns.lineplot(data=df.r)
    plt.xlabel('episodes')
    plt.ylabel('The cummulative return')
    plt.savefig(f"figures/reward.png")
    plt.close()

    # Visualize Controlled sliar Dynamics
    model = Algorithm.load(f'best_model/best_model.zip')
    state, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _, _ = eval_env.step(action)

    df = eval_env.dynamics
    # sns.lineplot(data=df, x='days', y='susceptible')
    plt.figure(figsize=(8,8))
    plt.subplot(5, 1, 1)
    plt.title(f"R = {df.rewards.sum():,.2f}")
    sns.lineplot(data=df, x='days', y='infected', color='r')
    plt.xticks(color='w')
    plt.subplot(5, 1, 2)
    sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre')
    plt.ylim([0.0, max(conf.sliar.nu_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 3)
    sns.lineplot(data=df, x='days', y='taus', color='b', drawstyle='steps-pre')
    plt.ylim([0.0, max(conf.sliar.tau_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 4)
    sns.lineplot(data=df, x='days', y='sigmas', color='orange', drawstyle='steps-pre')
    plt.ylim([0.0, max(conf.sliar.sigma_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 5)
    sns.lineplot(data=df, x='days', y='rewards', color='g')
    plt.savefig(f"figures/best.png")
    plt.close()

    best_checkpoint = ""
    max_val = -float('inf')
    for path in tqdm(os.listdir('checkpoints')):
        model = Algorithm.load(f'checkpoints/{path}')
        state, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(state)
            state, _, done, _, _ = eval_env.step(action)
        df = eval_env.dynamics

        cum_reward = df.rewards.sum()
        if cum_reward > max_val:
            max_val = cum_reward
            best_checkpoint = path

        plt.figure(figsize=(8,8))
        plt.subplot(5, 1, 1)
        plt.title(f"R = {df.rewards.sum():,.2f}")
        sns.lineplot(data=df, x='days', y='infected', color='r')
        plt.xticks(color='w')
        plt.subplot(5, 1, 2)
        sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre')
        plt.ylim([0.0, max(conf.sliar.nu_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 3)
        sns.lineplot(data=df, x='days', y='taus', color='b', drawstyle='steps-pre')
        plt.ylim([0.0, max(conf.sliar.tau_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 4)
        sns.lineplot(data=df, x='days', y='sigmas', color='orange', drawstyle='steps-pre')
        plt.ylim([0.0, max(conf.sliar.sigma_max * 1.1, 0.01)])
        plt.xticks(color='w')
        plt.subplot(5, 1, 5)
        sns.lineplot(data=df, x='days', y='rewards', color='g')
        plt.savefig(f"figures/{path.replace('.zip', '.png')}")
        plt.close()


    # Visualize Controlled sliar Dynamics
    model = Algorithm.load(f'checkpoints/{best_checkpoint}')
    state, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _, _ = eval_env.step(action)
    df = eval_env.dynamics
    # sns.lineplot(data=df, x='days', y='susceptible')
    plt.figure(figsize=(8,8))
    plt.subplot(5, 1, 1)
    plt.title(f"R = {df.rewards.sum():,.2f}")
    sns.lineplot(data=df, x='days', y='infected', color='r')
    plt.xticks(color='w')
    plt.subplot(5, 1, 2)
    sns.lineplot(data=df, x='days', y='nus', color='k', drawstyle='steps-pre')
    plt.ylim([0.0, max(conf.sliar.nu_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 3)
    sns.lineplot(data=df, x='days', y='taus', color='b', drawstyle='steps-pre')
    plt.ylim([0.0, max(conf.sliar.tau_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 4)
    sns.lineplot(data=df, x='days', y='sigmas', color='orange', drawstyle='steps-pre')
    plt.ylim([0.0, max(conf.sliar.sigma_max * 1.1, 0.01)])
    plt.xticks(color='w')
    plt.subplot(5, 1, 5)
    sns.lineplot(data=df, x='days', y='rewards', color='g')
    plt.savefig(f"figures/best_checkpoint.png")
    plt.close()
if __name__ == '__main__':
    main()