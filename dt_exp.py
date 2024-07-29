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
    max_ep_len = 1000
    #load data from csv file
    df = pd.read_csv(filename)
    
    DTconfig = {
        'model_type':'dt',
        'embed_dim':'128',
        'K':'20',
        'remove_act_embs':'true',
        'n_layer':'3',
        'n_head':'1',
        'activation_function':'gelu',
        'dropout':'0.1',
        'warmup_steps':'1000',
        'learning_rate':'1e-3',
        'weight_decay':'1e-4',
        'batch_size':'32',
        'max_iters':'10',
        'num_steps_per_iter':'1000'
    }
    
    
    
    #TODO create dataset
    episodes, timesteps, states, actions, rewards, RTGrewards,dones = [], [], [], [], [], []
    episodes = df['episode'].values
    timesteps = df['timestep'].values
    states = df['state'].values
    actions = df['action'].values
    rewards = df['reward'].values
    dones = df['done'].values
    RTGrewards = discount_cumsum(rewards, 1)
    
    
    model_type = DTconfig['model_type']
    model, optimizer, scheduler = get_model_optimizer(DTconfig, state_dim, action_dim, max_ep_len, device)
    print(f"{model_type}: #parameters = {sum(p.numel() for p in model.parameters())}")
    loss_fn = lambda a_hat, a: torch.mean((a_hat - a)**2)
    
    
    
    #***** ** utils ** *****
    def get_batch(batch_size=256, max_len=K):
        # Dynamically recompute p_sample if online training

        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            si = random.randint(0, traj['rewards'].shape[0] - 1)

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= max_ep_len] = max_ep_len-1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([np.zeros((1, max_len - tlen, state_dim)), s[-1]], axis=1)
            s[-1] = (s[-1] - state_mean) / state_std
            a[-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[-1]], axis=1)
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
            for _ in range(variant['num_eval_episodes']):
                with torch.no_grad():
                    ret, length = evaluate_episode_rtg(
                        env,
                        state_dim,
                        act_dim,
                        model,
                        max_ep_len=max_ep_len,
                        scale=scale,
                        target_return=target/scale,
                        mode=mode,
                        state_mean=state_mean,
                        state_std=state_std,
                        device=device
                    )
                returns.append(ret)
                lengths.append(length)
            reward_min = infos.REF_MIN_SCORE[f"{env_name}-{dataset}-v2"]
            reward_max = infos.REF_MAX_SCORE[f"{env_name}-{dataset}-v2"]
            return {
                f'target_{target}_return_mean': np.mean(returns),
                f'target_{target}_return_std': np.std(returns),
                f'target_{target}_length_mean': np.mean(lengths),
                f'target_{target}_d4rl_score': (np.mean(returns) - reward_min) * 100 / (reward_max - reward_min),  # compute the normalized reward, see https://github.com/Farama-Foundation/D4RL/blob/master/d4rl/offline_env.py#L71
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