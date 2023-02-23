import os
import hydra
from hydra.utils import instantiate
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

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
                            net_arch=[256, 128]
                        )
    model = PPO("MlpPolicy", train_env, verbose=0,
                policy_kwargs=policy_kwargs)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
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
                                             name_prefix='rl_model')
    callback = CallbackList([checkpoint_callback, eval_callback, ProgressBarCallback()])

    model.learn(total_timesteps=conf.n_steps, callback=callback)
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=100)
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
    state, _ = eval_env.reset()
    done = False
    while not done:
        action, _ = model.predict(state)
        state, _, done, _, _ = eval_env.step(action)

    df = eval_env.dynamics
    # sns.lineplot(data=df, x='days', y='susceptible')
    sns.lineplot(data=df, x='days', y='infected')
    ax2 = plt.twinx()
    sns.lineplot(data=df, x='days', y='nus', ax=ax2)
    sns.lineplot(data=df, x='days', y='taus', ax=ax2)
    sns.lineplot(data=df, x='days', y='sigmas', ax=ax2)
    plt.grid()
    plt.title(f"R = {df.rewards.sum():,.2f}")
    plt.savefig(f"figures/best.png")
    plt.close()

    for path in tqdm(os.listdir('checkpoints')):
        model = PPO.load(f'checkpoints/{path}')
        state, _ = eval_env.reset()
        done = False
        while not done:
            action, _ = model.predict(state)
            state, _, done, _, _ = eval_env.step(action)
        df = eval_env.dynamics
        fig, (a1, a2) = plt.subplots(2, 1)
        # sns.lineplot(ax=a1, data=df, x='days', y='susceptible')
        sns.lineplot(ax=a1, data=df, x='days', y='infected')
        ax2 = a1.twinx()
        sns.lineplot(data=df, x='days', y='nus', ax=ax2)
        sns.lineplot(data=df, x='days', y='taus', ax=ax2)
        sns.lineplot(data=df, x='days', y='sigmas', ax=ax2)
        plt.grid()
        sns.lineplot(ax=a2, data=df, x='days', y='rewards')
        plt.title(f"R = {df.rewards.sum():,.2f} ({int(path.split('_')[2]):,})")
        plt.savefig(f"figures/{path.replace('.zip', '.png')}")
        plt.close()
if __name__ == '__main__':
    main()