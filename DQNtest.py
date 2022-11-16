import os
import gym
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines.common.callbacks import BaseCallback
from stable_baselines.results_plotter import load_results, ts2xy
import logging
logging.getLogger('tensorflow').setLevel(logging.FATAL)
from IPython.display import clear_output

from stable_baselines import DQN
from stable_baselines.bench import Monitor
from stable_baselines.deepq.policies import MlpPolicy #deepq if DQN
from stable_baselines.common.evaluation import evaluate_policy
import time
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

#log_dir = "/home/zceened/ProjectFiles/DQN/Final/CLSE/4000/test/"
#log_dir1 = "/home/zceened/ProjectFiles/DQN/Final/CL/500/test/"
#os.makedirs(log_dir, exist_ok=True)
#os.makedirs(log_dir1, exist_ok=True)

topology_name = 'NSFNET'
k_paths = 5
with open(f"/content/XRL_MultiBand/optical-rl-gym/examples/topologies/NSFNET_chen_eon_5-paths_CLS.h5", 'rb') as f:
    topology = pickle.load(f)

# change number of nodes according to topology
node_request_probabilities = np.array([0.01801802, 0.04004004, 0.05305305, 0.01901902, 0.04504505, #add comma again
       0.02402402, 0.06706707, 0.08908909, 0.13813814, 0.12212212,
       0.07607608, 0.12012012, 0.01901902, 0.16916917])
env_args = dict(topology=topology, seed=11, allow_rejection=False,
               mean_service_holding_time=400, mean_service_inter_arrival_time=0.1, episode_length=50, j=1, node_request_probabilities=node_request_probabilities)
#env_args1 = dict(topology=topology, seed=11, allow_rejection=False,
  #             mean_service_holding_time=50, episode_length=50, j=1, node_request_probabilities=node_request_probabilities)
testing_env = gym.make('DeepRMSA-v0', **env_args)
#testing_env1 = gym.make('DeepRMSA-v0', **env_args1)
# Logs will be saved in log_dir/monitor.csv
#testing_env = Monitor(testing_env, log_dir + 'testing_upd', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))
#testing_env1 = Monitor(testing_env1, log_dir1 + 'testing_upd', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))
model = DQN.load("/content/XRL_MultiBand/DQN_results/NSFNET/CLS/4000/best.zip")
#model1 = DQN.load("/home/zceened/ProjectFiles/DQN/Final/CL/500/best.zip")
# Evaluate the agent
start = time.time()
mean_rewardD, std_rewardD = evaluate_policy(model, testing_env, n_eval_episodes=1000)
end = time.time()
print("Time required: ", (end-start))
print(f"mean_rewardbest:{mean_rewardD:.2f} +/- {std_rewardD:.2f}")

#mean_rewardD, std_rewardD = evaluate_policy(model1, testing_env1, n_eval_episodes=1000)
#print(f"mean_rewardbest:{mean_rewardD:.2f} +/- {std_rewardD:.2f}")
