import os
from datetime import datetime
from collections import deque
import torch
import torch.nn as nn
import random
import numpy as np
import torch.optim as optim
from time import sleep
from env import Environment, load_csv

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
    
def compute_acer_loss(policies, q_values, values, actions, rewards, retrace, masks, behavior_policies, gamma=0.99, truncation_clip=10, entropy_weight=0.0001):
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
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss
    
def off_policy_update(batch_size, replay_ratio=4):
    if batch_size > len(replay_buffer) + 1:
        return
    
    for _ in range(np.random.poisson(replay_ratio)):
        trajs = replay_buffer.sample(batch_size)
        state, action, reward, old_policy, mask = map(torch.stack, zip(*(map(torch.cat, zip(*traj)) for traj in trajs)))

        q_values = []
        values   = []
        policies = []

        for step in range(state.size(0)):
            policy, q_value, value = model(state[step])
            q_values.append(q_value)
            policies.append(policy)
            values.append(value)

        _, _, retrace = model(state[-1])
        retrace = retrace.detach()
        compute_acer_loss(policies, q_values, values, action, reward, retrace, mask, old_policy)
        
def transform_return_to_go(rewards):
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


if __name__ == "__main__": 
    print("setting training environment : ")
    vehicle_list = []
    default_path = "./highdensity/"
    for file in os.listdir(default_path):
        if file.endswith(".csv") and file.startswith("Trip"):
            vehicle_list.extend(load_csv(default_path + file))
    #vehicle_list = load_csv('vehicle_settings.csv')
    max_ep_len = len(vehicle_list)            # max timesteps in one episode
    gamma = 0.99                # discount factor
    lr_actor = 0.0003       # learning rate for actor network
    lr_critic = 0.001       # learning rate for critic network
    random_seed = 0       # set random seed
    max_training_timesteps = 200*len(vehicle_list)   # break from training loop if timeteps > max_training_timesteps
    print_freq = max_ep_len * 4     # print avg reward in the interval (in num timesteps)
    log_freq = max_ep_len * 2       # saving avg reward in the interval (in num timesteps)
    save_model_freq = max_ep_len * 4         # save model frequency (in num timesteps)
    capacity = 10000
    max_episode_length = 200



    env=Environment(vehicle_list)

    PREDIFINED_RESOURCE_BLOCK = 20

    # state space dimension, the extendral 3 dimensions here denotes depart time, arrival time and communication time
    state_dim = PREDIFINED_RESOURCE_BLOCK + 4
    env.set_state_dim(state_dim)
    # action space dimension denotes the probability of selection different time slots
    action_dim = PREDIFINED_RESOURCE_BLOCK
    env.set_action_dim(action_dim)
    ## Note : print/save frequencies should be > than max_ep_len

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


    checkpoint_path = directory + "ACER32_{}_{}_{}.pth".format('resource_allocation', random_seed, run_num_pretrained)
    print("save checkpoint path : " + checkpoint_path)

    #####################################################


    ############# print all hyperparameters #############

    print("--------------------------------------------------------------------------------------------")

    print("max training timesteps : ", max_training_timesteps)
    print("max timesteps per episode : ", max_episode_length)

    print("model saving frequency : " + str(save_model_freq) + " timesteps")
    print("log frequency : " + str(log_freq) + " timesteps")
    print("printing average reward over episodes in last : " + str(print_freq) + " timesteps")

    print("--------------------------------------------------------------------------------------------")

    print("state space dimension : ", state_dim)
    print("action space dimension : ", action_dim)

    print("--------------------------------------------------------------------------------------------")
 
    print("discount factor (gamma) : ", gamma)

    print("--------------------------------------------------------------------------------------------")

    print("optimizer learning rate (actor) : ", lr_actor)
    print("optimizer learning rate (critic) : ", lr_critic)

    print("--------------------------------------------------------------------------------------------")
    print("setting random seed to ", random_seed)
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    #####################################################

    print("============================================================================================")

    ################# training procedure ################
    # initialize ACER agent

    model = ActorCritic256(state_dim, action_dim).to(device)
    optimizer = optim.Adam(model.parameters())
    replay_buffer = EpisodicReplayMemory(capacity, max_episode_length)

    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)

    print("============================================================================================")


    # logging file
    log_f = open(log_f_name,"w+")
    log_f.write('episode,timestep,reward\n')

    #data file
    data_f = open(data_file_name,"w+")
    data_f.write('episode;timestep;state;action;reward;return\n')

    # printing and logging variables
    print_running_reward = 0
    print_running_episodes = 0

    log_running_reward = 0
    log_running_episodes = 0

    time_step = 0
    i_episode = 0
    num_steps    = 50
    log_interval = 10

    # start training loop
    while time_step <= max_training_timesteps:
        print("New training episode:")
        sleep(0.1) # we sleep to read the reward in console
        state = env.reset()
        current_ep_reward = 0
        q_values = []
        values   = []
        policies = []
        actions  = []
        rewards  = []
        masks    = []
        states = []
        RTGactions = []
        RTGreward = []
        dones = []

        for t in range(1, len(env.get_vehicle_list())+1):
            # select action with policy
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            policy, q_value, value = model(state)
        
            action = policy.multinomial(1)
            next_state, reward, done, info, RTGaction= env.step_1(action.item())
            #next_state, reward, done, info, RTGaction= env.step_withour_alpha(action.item())
            '''
            #sort the policy and select the first 10 action with the highest probability
            action = policy.topk(action_dim).indices
            action = action.numpy().tolist()[0]
            next_state, reward, done, info, action= env.step(action)
            action = torch.LongTensor([action]).unsqueeze(1).to(device)
            '''
        
            time_step +=1
            current_ep_reward += reward
    #        print("The current total episodic reward at timestep:", time_step, "is:", current_ep_reward)
    #        sleep(0.1) # we sleep to read the reward in console
            reward = torch.FloatTensor([reward]).unsqueeze(1).to(device)
            mask   = torch.FloatTensor(1 - np.float32([done])).unsqueeze(1).to(device)
            replay_buffer.push(state.detach(), action, reward, policy.detach(), mask, done)

            q_values.append(q_value)
            policies.append(policy)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            masks.append(mask)
            states.append(env.get_RTG_state())
            RTGactions.append(RTGaction)
            dones.append(done)
            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()
                print("Saving reward to csv file")
                sleep(0.1) # we sleep to read the reward in console
                log_running_reward = 0
                log_running_episodes = 0
            
            # printing average reward
            if time_step % print_freq == 0:

                # print average reward till last episode
                print_avg_reward = print_running_reward / print_running_episodes
                print_avg_reward = round(print_avg_reward, 2)
            
                print("Episode : {} \t\t Timestep : {} \t\t Average Reward : {}".format(i_episode, time_step, print_avg_reward))
                sleep(0.1) # we sleep to read the reward in console
                print_running_reward = 0
                print_running_episodes = 0
            
            # save model weights
            if time_step % save_model_freq == 0:
                print("--------------------------------------------------------------------------------------------")
                print("saving model at : " + checkpoint_path)
                sleep(0.1) # we sleep to read the reward in console
                torch.save(model.state_dict(), checkpoint_path)
                print("model saved")
                print("--------------------------------------------------------------------------------------------")
            state = next_state
        
            # break; if the episode is over
            if done:
                current_round_peroid, longest_vehicle_depart = env.statistic_scheduled_vehicles()
                print("The current round peroid is: ", current_round_peroid)
                print("The longest vehicle depart time is: ", longest_vehicle_depart)
                break
        RTGreward = transform_return_to_go(rewards)
        #merge action, state, and RTGreward into one array, the format is [RTGreward,state, action]
        trajectory = []

        for i in range(len(RTGreward)):
            tsp = i
            temp_state = states[i]
            temp_action = RTGactions[i]
            temp_reward = rewards[i].item()
            temp_RTGreward = RTGreward[i].item()
            #write the trajectory to the data file
            #data_f.write('{};{};{};{};{};{};{}\n'.format(i_episode, tsp, temp_state, temp_action, temp_reward, temp_RTGreward,dones[i]))

        #write the trajectory to the data file
   
        trajectory = []
        next_state = torch.FloatTensor(state).unsqueeze(0).to(device)
        _, _, retrace = model(next_state)
        retrace = retrace.detach()
        compute_acer_loss(policies, q_values, values, actions, rewards, retrace, masks, policies)
    
    
        off_policy_update(128)
    
        print_running_reward += current_ep_reward
        print_running_episodes += 1

        log_running_reward += current_ep_reward
        log_running_episodes += 1

        i_episode += 1
        time_step += num_steps
    

    log_f.close()
    data_f.close()
    ################################ End of Part II ################################

    print("============================================================================================")


