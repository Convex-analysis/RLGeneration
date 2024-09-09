import os
import argparse
from datetime import datetime
import random
import pickle
import yaml


import numpy as np
import pandas as pd
import torch
from env import Environment, load_csv


from util.trainer import Trainer, evaluate_episode_rtg, get_model_optimizer
from util.utils import set_seed, discount_cumsum, get_outdir, update_summary


SEED = 1
STATE = 24
ACTION = 20
Vlist = 'vehicle_settings.csv'

def main(filename):
    vehicle_list = load_csv(Vlist)
    env = Environment(vehicle_list)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'  #variant.get('device', 'cuda')
    set_seed(SEED)
    state_dim = STATE
    action_dim = ACTION
    max_ep_len = 1000
    env_targets = [20, 50, 100]
    scale = 1000
    #load data from csv file
    df = pd.read_csv(filename, sep=';')
    
    DTconfig = {
        'model_type':'dt',
        'embed_dim':128,
        'K':20,
        'remove_act_embs':'true',
        'n_layer':3,
        'n_head':1,
        'activation_function':'gelu',
        'dropout':0.1,
        'warmup_steps':1000,
        'learning_rate':1e-3,
        'weight_decay':1e-4,
        'batch_size':32,
        'max_iters':10,
        'num_steps_per_iter':1000,
        'num_eval_episodes':100
    }
    
    
    
    #TODO create dataset
    episodes, timesteps, states, actions, rewards, RTGrewards,dones = [], [], [], [], [], [],[]
    episodes = df['episode'].values
    timesteps = df['timestep'].values
    states = df['state'].values
    actions = df['action'].values
    rewards = df['reward'].values
    dones = df['done'].values
    RTGrewards = discount_cumsum(rewards, 1)
    
    
    trajectories = []
    trajectory = {
        'episode':'',
        'state': [],
        'action': [],
        'reward': [],
        }
    former_episode = episodes[0]
    for i in range(len(episodes)):
        episode = episodes[i]
        #if new episode, add former trajectory to trajectories and create new trajectory
        if episode != former_episode:
            trajectories.append(trajectory)
            trajectory = {
                'episode':'',
                'state': [],
                'action': [],
                'reward': [],
                }
        #add data to trajectory
        trajectory['episode'] = episode
        trajectory['state'].append(states[i])
        trajectory['action'].append(actions[i])
        trajectory['reward'].append(rewards[i])
        #update former episode
        former_episode = episode
    
    #save all trajectory information into separate lists    
    path_state, traj_lens, path_returns = [], [], []    
    for path in trajectories:
        path_state.append(path['state'])
        traj_lens.append(len(path['state']))
        path_returns.append(np.sum(path['reward']))
    
    traj_lens, path_returns = np.array(traj_lens), np.array(path_returns)

    # normalize returns
    path_state = np.concatenate(path_state, axis=0)
    #seprate the state into resourcepool, vehicle depart time, vehicle arrival time, communication time and alpha
    for i, state in enumerate(path_state):
        state = state.split(',')
        if len(state) > 5:
            resourcepool = state[0:-4]
            #delete the first character '['
            resourcepool[0] = resourcepool[0][1:]
            #transform the resourcepool into a string
            resourcepool = ','.join(resourcepool)
            resourcepool.replace(', ', ',')
            #let the string in [] to be a array
            resourcepool = resourcepool.split(']')
            resourcepool = [x.split('[')[1] for x in resourcepool if x != '']
            resourcepool = [x.split(',') for x in resourcepool]
            resourcecap = [int(x[1])-int(x[0]) for x in resourcepool]
                
                
            vehicle_depart_time = state[-4].strip()
            length = vehicle_depart_time.isdigit()
            vehicle_arrival_time = state[-3].strip()
            communication_time = state[-2].strip()
            alpha = state[-1].split(']')[0].strip()
            vehicle_depart_time, vehicle_arrival_time, communication_time, alpha = float(vehicle_depart_time), float(vehicle_arrival_time), float(communication_time), float(alpha)
            #satate = resourcepool + vehicle_depart_time + vehicle_arrival_time + communication_time + alpha
            
        
    
    #state_mean = np.mean(path_state, axis=0)
    #state_std = np.std(path_state, axis=0)
    num_timesteps = sum(traj_lens)
    

    sorted_index = np.argsort(path_returns) # low to high
    num_trajectories = 1
    timestep = traj_lens[sorted_index[-1]]# find the trajectory with the highest return and get its length
    # find the number of trajectories that fit in the num_timesteps
    index = len(trajectories) - 2 #the index here denote the index of the trajectory with the next highest return
    # iterate through the trajectories in descending order of return
    while index >= 0 and timestep + traj_lens[sorted_index[index]] <= num_timesteps:
        timestep += traj_lens[sorted_index[index]]
        num_trajectories += 1
        index -= 1
    # get the trajectories with the highest returns
    sorted_index = sorted_index[-num_trajectories:]
    
    
    prob_sample = traj_lens[sorted_index] / sum(traj_lens[sorted_index])
    model_type = DTconfig['model_type']
    start_time = datetime.now().replace(microsecond=0)
    start_time_str = start_time.strftime("%y-%m-%d-%H-%M-%S")
    environment = 'DT'
    save_model_name = model_type + "_" + environment + ".cpt"


    # setup output dir and yaml file
    output_dir = get_outdir('./output/train', 'DT')
    args_dir = os.path.join(output_dir, 'args.yaml')
    with open(args_dir, 'w') as f:
        f.write(yaml.safe_dump(DTconfig, default_flow_style=False))


    
    model, optimizer, scheduler = get_model_optimizer(DTconfig, state_dim, action_dim, max_ep_len, device)
    print(f"{model_type}: #parameters = {sum(p.numel() for p in model.parameters())}")
    loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)
    
    
    
    #***** ** utils ** *****
    def get_batch(batch_size=256, max_len=DTconfig['K']):
        # Dynamically recompute p_sample if online training

        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=prob_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_index[batch_inds[i]])]
            si = random.randint(0, traj['reward'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['state'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['action'][si:si + max_len].reshape(1, -1, action_dim))
            r.append(traj['reward'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['reward'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            #s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, action_dim)) * -10., a[-1]], axis=1)
            r[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[-1]], axis=1)
            d[-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[-1]], axis=1)
            rtg[-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[-1]], axis=1) / scale
            timesteps[-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        return s, a, r, d, rtg, timesteps, mask

    def eval_episodes(target):
        def fn(model):
            returns, lengths = [], []
            for _ in range(DTconfig['num_eval_episodes']):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        action_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target/scale,
                        mode=DTconfig['model_type'],
                        device=device
                    )
                returns.append(ret)
                lengths.append(length)
            #reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v2"]
            #reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v2"]
            return {
                f'target_{target}_return_mean': np.mean(returns),
                f'target_{target}_return_std': np.std(returns),
                f'target_{target}_length_mean': np.mean(lengths),
                #f'target_{target}_d4rl_score': (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),  # compute the normalized reward, see https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py#L71
            }
        return fn
    #***** ***** ***** *****
    
    
    
    trainer = Trainer(
        model_type=model_type,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        batch_size=DTconfig['batch_size'],
        get_batch=get_batch,
        loss_fn=loss_fn,
        eval_fns=[eval_episodes(tar) for tar in env_targets],
    )

    n_iter = 0
    try:
        for _iter in range(DTconfig['max_iters']):
            outputs = trainer.train_iteration(num_steps=DTconfig['num_steps_per_iter'], iter_num=_iter+1, print_logs=True)
            if output_dir is not None:
                update_summary(
                    _iter,
                    outputs,
                    filename=os.path.join(output_dir, 'summary.csv'),
                    args_dir=args_dir,
                    write_header=_iter == 0
                    )
            n_iter += 1
    except KeyboardInterrupt:
        pass

    save_state = {
        'epoch': n_iter+1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(save_state, os.path.join(output_dir, save_model_name))
    
    
if __name__ == '__main__':
    filename = 'ACER_files/data/18_data.csv'
    main(filename)