import os
import gym
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import load_results, ts2xy
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from IPython.display import clear_output
from stable_baselines3 import DQN
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import time
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_param_importances
import kaleido


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve training reward
            x, y = ts2xy(load_results(self.log_dir), 'timesteps')
            if len(x) > 0:
                 # Mean training reward over the last 100 episodes
                mean_reward = np.mean(y[-100:])
                if self.verbose > 0:
                    print("Num timesteps: {} - ".format(self.num_timesteps),end="")
                    print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))
                  # New best model, you could save the agent here
                if mean_reward > self.best_mean_reward:
                    self.best_mean_reward = mean_reward
                    # Example for saving best model
                    if self.verbose > 0:
                        print("Saving new best model to {}".format(self.save_path))
                        self.model.save(self.save_path)
                if self.verbose > 0:
                    clear_output(wait=True)

        return True

# Specify hyperparameters of DQN being optimized
def optimize_dqn(trial):
    gamma=trial.suggest_float('gamma', 0.8, 0.9999)
    exploration_fraction=trial.suggest_float('exploration_fraction', 0.05, 0.3)
    learning_rate=trial.suggest_float('learning_rate', 1e-6, 1e-4)
    batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    buffer_size=trial.suggest_categorical('buffer_size', [int(1e4), int(1e5), int(1e6), int(1e7)])
    tau=trial.suggest_float('tau', 0.01, 1)

    return {'gamma': gamma,
        'exploration_fraction': exploration_fraction,
        'learning_rate': learning_rate,
        'batch_size': batch_size,
        'buffer_size': buffer_size,
        'tau': tau}

def objective(trial):
    try:
        hyperparameters = optimize_dqn(trial)
        k_paths = 5
        with open(f'/home/zceehbi/XRL_MultiBand/optical-rl-gym/examples/topologies/German_5-paths_CLSE.h5', 'rb') as f:
            topology = pickle.load(f)
        topology_name = 'German'
        node_request_probabilities = np.ones(17)/17
        load = 4000
        holding_time = 400

        log_dir = "/home/zceehbi/XRL_MultiBand/DQN_results/Optuna/German/{}/tmp1/".format(load)
        os.makedirs(log_dir, exist_ok=True)
        callback = SaveOnBestTrainingRewardCallback(check_freq=100, log_dir=log_dir)
        env_args = dict(topology=topology, seed=10, allow_rejection=False,
                    mean_service_holding_time=holding_time, episode_length=50, j=1, node_request_probabilities=node_request_probabilities)
        training_env = gym.make('DeepRMSA-v0', **env_args)
        training_env = Monitor(training_env, log_dir + '100ktraining', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))

        # Use hyperparamters from optuna
        policy_kwargs = {'net_arch': [128] * 4}
        model = DQN('MlpPolicy', training_env, verbose=0, tensorboard_log="/home/zceehbi/XRL_MultiBand/tb/Optuna/German_DQN/", policy_kwargs=policy_kwargs, **hyperparameters)
        model.learn(total_timesteps=100000, callback=callback)

        env_args['seed'] = 11
        testing_env = gym.make('DeepRMSA-v0', **env_args)
        testing_env = Monitor(testing_env, log_dir + '100ktesting', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))

        mean_reward, std_reward = evaluate_policy(model, testing_env, n_eval_episodes=1000)

        model.save("/home/zceehbi/XRL_MultiBand/DQN_results/Optuna/German/models/tmp1/trial_{}_best" .format(trial.number))

        return mean_reward

    # return a huge negative reward when there is error
    except Exception as e:
        return -1000

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20)

print('Best hyperparameters:')
print(study.best_params)
print('Best trial:')
print(study.best_trial)

figure1 = plot_parallel_coordinate(study)
figure2 = plot_param_importances(study)
figure3 = plot_optimization_history(study)
figure1.write_image(file='/home/zceehbi/Results/Optuna/Figure1.png', format='png')
figure2.write_image(file='/home/zceehbi/Results/Optuna/Figure2.png', format='png')
figure3.write_image(file='/home/zceehbi/Results/Optuna/Figure3.png', format='png')