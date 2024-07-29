import os
import argparse
from datetime import datetime
import random
import pickle
import yaml


import numpy as np
import pandas as pd
import torch


from util.trainer import Trainer, get_env_info, evaluate_episode_rtg, get_model_optimizer
from util.utils import set_seed, discount_cumsum, get_outdir, update_summary


SEED = 1
STATE = 24
ACTION = 20


def main(filename):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  #variant.get('device', 'cuda')
    set_seed(SEED)
    state_dim = STATE
    action_dim = ACTION
    
    #load data from csv file
    df = pd.read_csv(filename)
    
    #TODO create dataset
    episodes, timesteps, states, actions, rewards, RTGrewards,dones = [], [], [], [], [], []
    episodes = df['episode'].values
    timesteps = df['timestep'].values
    states = df['state'].values
    actions = df['action'].values
    rewards = df['reward'].values
    dones = df['done'].values
    RTGrewards = discount_cumsum(rewards, 1)
    