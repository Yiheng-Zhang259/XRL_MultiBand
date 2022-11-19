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

topology_name = 'German'
k_paths = 5
with open(f"/content/XRL_MultiBand/optical-rl-gym/examples/topologies/German_5-paths_CLSE.h5", 'rb') as f:
    topology = pickle.load(f)

# change number of nodes according to topology
node_request_probabilities = np.ones(17)/17

# list to store the training results
rewards=[]

for ht in [50, 100, 200, 400, 600, 800, 1000, 1200]:
    env_args = dict(topology=topology, seed=11, allow_rejection=False,
                    mean_service_holding_time=ht, mean_service_inter_arrival_time=0.1, episode_length=50, j=1, node_request_probabilities=node_request_probabilities)
    #env_args1 = dict(topology=topology, seed=11, allow_rejection=False,
                #   mean_service_holding_time=ht, episode_length=50, j=1, node_request_probabilities=node_request_probabilities)
    testing_env = gym.make('DeepRMSA-v0', **env_args)
    #testing_env1 = gym.make('DeepRMSA-v0', **env_args1)
    # Logs will be saved in log_dir/monitor.csv
    #testing_env = Monitor(testing_env, log_dir + 'testing_upd', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))
    #testing_env1 = Monitor(testing_env1, log_dir1 + 'testing_upd', info_keywords=('episode_service_blocking_rate','episode_bit_rate_blocking_rate'))
    model = DQN.load("/content/XRL_MultiBand/DQN_results/German/CLSE/{}/tmp1/best.zip".format(ht/0.1))
    #model1 = DQN.load("/home/zceened/ProjectFiles/DQN/Final/CL/500/best.zip")
    # Evaluate the agent
    start = time.time()
    mean_rewardD, std_rewardD = evaluate_policy(model, testing_env, n_eval_episodes=1000)
    end = time.time()
    print("Time required: ", (end-start))
    print(f"mean_rewardbest:{mean_rewardD:.2f} +/- {std_rewardD:.2f}")
    rewards.append(mean_rewardD)

    #mean_rewardD, std_rewardD = evaluate_policy(model1, testing_env1, n_eval_episodes=1000)
    #print(f"mean_rewardbest:{mean_rewardD:.2f} +/- {std_rewardD:.2f}")

#save the reward data
with open("German_CLSE_100k_testing", "wb") as fp:
    pickle.dump(rewards, fp)
