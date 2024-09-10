import os
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
import pandas as pd
from time import sleep
from env import Environment, load_csv

PREDIFINED_RESOURCE_BLOCK = 20
MAX_EPISODE_LENGTH = 200

device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

class EpisodicReplayMemory(object):
    def __init__(self, capacity, max_episode_length):
        self.num_episodes = capacity // max_episode_length
        self.buffer = deque(maxlen=self.num_episodes)
        self.buffer.append([])
        self.position = 0
        
    def push(self, state, action, reward, policy, mask, done):
        self.buffer[self.position].append((state, action, reward, policy, mask))
        if done:
            self.buffer.append([])
            self.position = min(self.position + 1, self.num_episodes - 1)
            
    def sample(self, batch_size, max_len=None):
        min_len = 0
        while min_len == 0:
            rand_episodes = random.sample(self.buffer, batch_size)
            min_len = min(len(episode) for episode in rand_episodes)
            
        if max_len:
            max_len = min(max_len, min_len)
        else:
            max_len = min_len
            
        episodes = []
        for episode in rand_episodes:
            if len(episode) > max_len:
                rand_idx = random.randint(0, len(episode) - max_len)
            else:
                rand_idx = 0

            episodes.append(episode[rand_idx:rand_idx+max_len])
            
        return list(map(list, zip(*episodes)))
    
    def __len__(self):
        return len(self.buffer)

class ActorCritic32(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=32):
        super(ActorCritic32, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
        
        
    def forward(self, x):
        policy  = self.actor(x).clamp(max=1-1e-20)
        q_value = self.critic(x)
        value   = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
    
class ActorCritic64(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=64):
        super(ActorCritic64, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
        
        
    def forward(self, x):
        policy  = self.actor(x).clamp(max=1-1e-20)
        q_value = self.critic(x)
        value   = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
    
    
class ActorCritic256(nn.Module):
    """
    A class representing an Actor-Critic network with a hidden size of 256.

    Args:
        num_inputs (int): The number of input features.
        num_actions (int): The number of possible actions.
        hidden_size (int, optional): The size of the hidden layer. Defaults to 256.
    """
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCritic256, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
        
        
    def forward(self, x):
        """
        Forward pass of the Actor-Critic network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            tuple: A tuple containing the policy, q_value, and value tensors.
        """
        policy  = self.actor(x).clamp(max=1-1e-20)
        q_value = self.critic(x)
        value   = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
    
    
class ActorCriticTest(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size=256):
        super(ActorCriticTest, self).__init__()
        
        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions),
            nn.Softmax(dim=1)
        )
        
        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, num_actions)
        )
        
    def forward(self, x):
        policy  = self.actor(x).clamp(max=1-1e-20)
        q_value = self.critic(x)
        value   = (policy * q_value).sum(-1, keepdim=True)
        return policy, q_value, value
    


class simulation():
    def __init__(self):
        self.max_ep_len = MAX_EPISODE_LENGTH           # max timesteps in one episode
        self.gamma = 0.99                # discount factor
        self.lr_actor = 0.0003       # learning rate for actor network
        self.lr_critic = 0.001       # learning rate for critic network
        self.random_seed = 0       # set random seed
        self.max_training_timesteps = 100*self.max_ep_len   # break from training loop if timeteps > max_training_timesteps
        self.print_freq = self.max_ep_len * 4     # print avg reward in the interval (in num timesteps)
        self.log_freq = self.max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
        self.save_model_freq = self.max_ep_len * 4         # save model frequency (in num timesteps)
        self.capacity = 10000
        self.max_episode_length = 200
        
        
        self.env= None
        self.model = None
        self.optimizer = None
        self.replay_buffer = None
        
        self.log_f = None
        self.data_f = None
        
    
    def initialize_environment(self, vehicle_list):
        env=Environment(vehicle_list)
        # state space dimension, the extendral 3 dimensions here denotes depart time, arrival time and communication time
        state_dim = PREDIFINED_RESOURCE_BLOCK + 4
        env.set_state_dim(state_dim)
        # action space dimension denotes the probability of selection different time slots
        action_dim = PREDIFINED_RESOURCE_BLOCK
        env.set_action_dim(action_dim)
        ## Note : print/save frequencies should be > than max_ep_len
        self.env = env
        
    def initialize_ACER_agent(self):
        model = ActorCritic256(self.env.get_state_dim(), self.env.get_action_dim()).to(device)
        optimizer = optim.Adam(model.parameters())
        replay_buffer = EpisodicReplayMemory(self.capacity, self.max_episode_length)
        self.model = model
        self.optimizer = optimizer
        self.replay_buffer = replay_buffer
    
    
    def log_file_saving_profile(self):
        ###################### saving files ######################

        #### saving files for multiple runs are NOT overwritten

        log_dir = "ACER_files"
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        log_dir = log_dir + '/' + 'resource_allocation' + '/'+ 'stability' + '/'
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        generate_data_dir = 'ACER_files' + '/' + 'data'
        if not os.path.exists(generate_data_dir):
            os.makedirs(generate_data_dir)   

        #### get number of saving files in directory
        run_num = 0
        current_num_files = next(os.walk(log_dir))[2]
        run_num = len(current_num_files)
        #### create new saving file for each run
        data_file_name = generate_data_dir + '/' + str(run_num) + '_data.csv'
        #### create new saving file for each run 
        log_f_name = log_dir + '/ACER_' + 'resource_allocation' + "_log_" + str(run_num) + ".csv"


        print("current logging run number for " + 'resource_allocation' + " : ", run_num)
        print("logging at : " + log_f_name)

        #####################################################

        ################### checkpointing ###################

        run_num_pretrained = 0      #### change this to prevent overwriting weights in same env_name folder

        directory = "ACER_preTrained"
        if not os.path.exists(directory):
            os.makedirs(directory)

        directory = directory + '/' + 'resource_allocation' + '/' 
        if not os.path.exists(directory):
            os.makedirs(directory)


        checkpoint_path = directory + "ACER32_{}_{}_{}.pth".format('resource_allocation', self.random_seed, run_num_pretrained)
        print("save checkpoint path : " + checkpoint_path)
        
        return log_f_name, data_file_name, checkpoint_path

    def print_training_profile(self):
        #####################################################


        ############# print all hyperparameters #############

        print("--------------------------------------------------------------------------------------------")

        print("max training timesteps : ", self.max_training_timesteps)
        print("max timesteps per episode : ", self.max_episode_length)

        print("model saving frequency : " + str(self.save_model_freq) + " timesteps")
        print("log frequency : " + str(self.log_freq) + " timesteps")
        print("printing average reward over episodes in last : " + str(self.print_freq) + " timesteps")

        print("--------------------------------------------------------------------------------------------")

        print("state space dimension : ", self.env.get_state_dim())
        print("action space dimension : ", self.env.get_action_dim())

        print("--------------------------------------------------------------------------------------------")
    
        print("discount factor (gamma) : ", self.gamma)

        print("--------------------------------------------------------------------------------------------")

        print("optimizer learning rate (actor) : ", self.lr_actor)
        print("optimizer learning rate (critic) : ", self.lr_critic)

        print("--------------------------------------------------------------------------------------------")
        print("setting random seed to ", self.random_seed)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)


    def training_episode(self, checkpoint_path, time_step=0, i_episode=0, num_steps=50,print_running_reward=0, print_running_episodes=0, log_running_reward=0, log_running_episodes=0):
        # printing and logging variables
        
        state = self.env.reset()
        current_ep_reward = 0
        q_values = []
        values   = []
        policies = []
        actions  = []
        rewards  = []
        masks    = []
        dones = []
        
        current_round_peroid, longest_vehicle_depart, scheduled_number = 0, 0, 0

        for t in range(1, len(self.env.get_vehicle_list())+1):
            # select action with policy
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, q_value, value = self.model(state)
        
            action = policy.multinomial(1)
            next_state, reward, done, info, RTGaction= self.env.step_1(action.item())
            #next_state, reward, done, info, RTGaction= env.step_withour_alpha(action.item())       
            time_step +=1
            current_ep_reward += reward
            reward = torch.FloatTensor([reward]).unsqueeze(1).to(device)
            mask   = torch.FloatTensor(1 - np.float32([done])).unsqueeze(1).to(device)
            self.replay_buffer.push(state.detach(), action, reward, policy.detach(), mask, done)

            q_values.append(q_value)
            policies.append(policy)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            masks.append(mask)
            dones.append(done)
            # log in logging file
            if time_step % self.log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                self.log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                self.log_f.flush()
                print("Saving reward to csv file")
                sleep(0.1) # we sleep to read the reward in console
                log_running_reward = 0
                log_running_episodes = 0
            
            # printing average reward
            if time_step % self.print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
            
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                sleep(0.1) # we sleep to read the reward in console
                print_running_reward = 0
                print_running_episodes = 0
            
            # save model weights
            if time_step % self.save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                sleep(0.1) # we sleep to read the reward in console
                torch.save(self.model.state_dict(), checkpoint_path)
                print("model saved")
                print("--------------------------------------------------------------------------------------------")
            state = next_state
        
            # break; if the episode is over
            if done:
                current_round_peroid, longest_vehicle_depart, scheduled_number = self.env.statistic_scheduled_vehicles()
                break
        next_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        _, _, retrace = self.model(next_state)
        retrace = retrace.detach()
        self.compute_acer_loss(policies, q_values, values, actions, rewards, retrace, masks, policies)
    
    
        self.off_policy_update(128)
    
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        time_step += num_steps
        
        return time_step, i_episode, print_running_reward, print_running_episodes, log_running_reward, log_running_episodes,current_round_peroid, longest_vehicle_depart, scheduled_number


        
    def compute_acer_loss(self,policies, q_values, values, actions, rewards, retrace, masks, behavior_policies, gamma=0.99, truncation_clip=10, entropy_weight=0.0001):
        loss = 0
    
        for step in reversed(range(len(rewards))):
            importance_weight = policies[step].detach() / behavior_policies[step].detach()

            retrace = rewards[step] + gamma * retrace * masks[step]
            advantage = retrace - values[step]

            log_policy_action = policies[step].gather(1, actions[step]).log()
            truncated_importance_weight = importance_weight.gather(1, actions[step]).clamp(max=truncation_clip)
            actor_loss = -(truncated_importance_weight * log_policy_action * advantage.detach()).mean(0)

            correction_weight = (1 - truncation_clip / importance_weight).clamp(min=0)
            actor_loss -= (correction_weight * policies[step].log() * (q_values[step] - values[step]).detach()).sum(1).mean(0)
            
            entropy = entropy_weight * -(policies[step].log() * policies[step]).sum(1).mean(0)

            q_value = q_values[step].gather(1, actions[step])
            critic_loss = ((retrace - q_value) ** 2 / 2).mean(0)

            truncated_rho = importance_weight.gather(1, actions[step]).clamp(max=1)
            retrace = truncated_rho * (retrace - q_value.detach()) + values[step].detach()
            
            loss += actor_loss + critic_loss - entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss
    
    def off_policy_update(self,batch_size, replay_ratio=4):
        if batch_size > len(self.replay_buffer) + 1:
            return
        
        for _ in range(np.random.poisson(replay_ratio)):
            trajs = self.replay_buffer.sample(batch_size)
            state, action, reward, old_policy, mask = map(torch.stack, zip(*(map(torch.cat, zip(*traj)) for traj in trajs)))

            q_values = []
            values   = []
            policies = []

            for step in range(state.size(0)):
                policy, q_value, value = self.model(state[step])
                q_values.append(q_value)
                policies.append(policy)
                values.append(value)

            _, _, retrace = self.model(state[-1])
            retrace = retrace.detach()
            self.compute_acer_loss(policies, q_values, values, action, reward, retrace, mask, old_policy)
            
    def transform_return_to_go(self,rewards):
        """
        Transforms a list of rewards into a list of return-to-go rewards.

        Args:
            rewards (list): A list of rewards.

        Returns:
            list: A list of return-to-go rewards, where each element is the sum of rewards[i:].

        """
        RTGreward = []
        for i in range(len(rewards)):
            RTGreward.append(sum(rewards[i:]))
        return RTGreward
        
        
    
    def simulate_process(self,filename="vehicle_settings.csv"):
        self.initialize_environment(load_csv(filename))
        self.initialize_ACER_agent()
        log_f_name, data_file_name, checkpoint_path = self.log_file_saving_profile()
        self.print_training_profile()
        start_time = datetime.now().replace(microsecond=0)
        print("Started training at (GMT) : ", start_time)

        print("============================================================================================")


        # logging file
        self.log_f = open(log_f_name,"w+")
        self.log_f.write('episode,timestep,reward\n')
        
        print_running_reward = 0
        print_running_episodes = 0

        log_running_reward = 0
        log_running_episodes = 0

        time_step = 0
        i_episode = 0
        num_steps    = 50

        longest_vehicle_depart_list = []
        round_peroid_list = []
        scheduled_number_list = []

        # start training loop
        while time_step <= self.max_training_timesteps:
            # train for an episode
            time_step, i_episode, print_running_reward, print_running_episodes, log_running_reward, log_running_episodes,current_round_peroid, longest_vehicle_depart, scheduled_number = self.training_episode(checkpoint_path, time_step, i_episode, num_steps,print_running_reward, print_running_episodes, log_running_reward, log_running_episodes)
            # store the results
            longest_vehicle_depart_list.append(longest_vehicle_depart)
            round_peroid_list.append(current_round_peroid)
            scheduled_number_list.append(scheduled_number)

        self.log_f.close()
        return longest_vehicle_depart_list, round_peroid_list,scheduled_number_list


if __name__ == "__main__": 
    SM = simulation()
    filelist = []
    df = pd.DataFrame(columns = ["round", "longest_vehicle_depart", "round_peroid", "scheduled_number"])
    round_count = 0
    default_path = "./highdensity/"
    for file in os.listdir(default_path):
        if file.endswith(".csv") and file.startswith("Trip"):
            filelist.append(file)
    for file in filelist:
        longest_vehicle_depart_list, round_peroid_list, scheduled_number_list = [], [], []
        print("Loading file: ", file)
        longest_vehicle_depart_list, round_peroid_list, scheduled_number_list = SM.simulate_process(default_path + file)
        round_count += 1
        #delete the zero value in the list
        longest_vehicle_depart_list = [x for x in longest_vehicle_depart_list if x != 0]
        round_peroid_list = [x for x in round_peroid_list if x != 0]
        scheduled_number_list = [x for x in scheduled_number_list if x != 0]
        
        avg_longest_vehicle_depart = sum(longest_vehicle_depart_list) / len(longest_vehicle_depart_list)
        avg_round_peroid = sum(round_peroid_list) / len(round_peroid_list)
        avg_scheduled_number = sum(scheduled_number_list) / len(scheduled_number_list)
        
        log = [round_count, avg_longest_vehicle_depart, avg_round_peroid, avg_scheduled_number]
        #add the log to the dataframe
        df.loc[len(df)] = log
    
    df.to_csv("result.csv", index=False)


