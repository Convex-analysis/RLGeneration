import os
import heapq
import math
import pandas as pd
import numpy as np
from collections import deque

# Constants for the one round time of FL corresponding to the simulation time of SUMO
ONE_ROUND_TIME = 1000
REQUIRED_EPOCHS = 10
AVAILABLE_BANDWIDTH = 3

def load_csv(file):
    df = pd.read_csv(file)
    df = df.sort_values(by = "depart")
    vehicle_list = []
    for index, row in df.iterrows():
        vehicle = Vehicle(row["id"], row["depart"], row["duration"], row["arrival"], row["train"], row["communication"], row["alpha"])
        vehicle_list.append(vehicle)
    return vehicle_list


class Vehicle():
    def __init__(self, id, depart, duration, arrival, train, communication, alpha):
        self.id = id
        self.depart = depart
        self.duration = duration
        self.arrival = arrival
        self.train = train
        self.communication = communication
        self.alpha = alpha
        
    #generate the corresponding get and set methods
    def get_id(self):
        return self.id
    def get_depart(self):
        return self.depart
    def get_duration(self):
        return self.duration
    def get_arrival(self):
        return self.arrival
    def get_train(self):
        return self.train
    def get_communication(self):
        return self.communication
    def get_alpha(self):
        return self.alpha
        
        

class Environment():
    def __init__(self, vehicle_list):
        self.vehicle_list = vehicle_list
        '''
        Generate the resource pool as a dictionary with three keys: "1", "2", "3"
        Specifically, the architecture of the resource pool is as follows:
        bandwidth index|avaliable reource block list|
        1              |{0,1000}|
        2              |{0,1000}|
        3              |{0,1000}|
        and when the vehicle successfully be scheduled, we update the resource pool by removing the corresponding resource block from the available resource block list as follows:
        bandwidth index|avaliable reource block list|
        1              |{0,50},{100, 1000}|          #the vehicle uses the resource block from 50 to 100
        2              |{0,1000}|
        3              |{0,1000}|
        '''
        
        ResourcePool = {}
        for i in range(1, AVAILABLE_BANDWIDTH+1):
            ResourcePool[str(i)] = [[0, ONE_ROUND_TIME]]
        self.resourcepool = ResourcePool
        self.time = 0
        self.vehicle_queue = deque(self.vehicle_list)
        self.scheduled_vehicle = []
        self.state_dim = 0
        self.action_dim = 0
        self.current_state = np.array([])
    
    def get_time(self):
        return self.time
    
    def set_time(self, time):
        self.time = time
        
    def get_vehicle_list(self):
        return self.vehicle_list
    
    def get_vehicle_queue(self):
        return self.vehicle_queue
    
    def set_vehicle_queue(self, vehicle_queue):
        self.vehicle_queue = vehicle_queue
    
    def get_scheduled_vehicle(self):
        return self.scheduled_vehicle
    
    def set_scheduled_vehicle(self, scheduled_vehicle):
        self.scheduled_vehicle = scheduled_vehicle
    
    def get_resourcepool(self):
        return self.resourcepool
    
    def set_resourcepool(self, resourcepool):
        self.resourcepool = resourcepool
        
    def get_state_dim(self):
        return self.state_dim
    
    def set_state_dim(self, state_dim):
        self.state_dim = state_dim
        
    def get_action_dim(self):
        return self.action_dim
    
    def set_action_dim(self, action_dim):
        self.action_dim = action_dim    
    
    def get_current_vehicle(self):
        #get the current vehicle
        current_vehicle = self.vehicle_queue[0]
        #update the vehicle queue
        self.vehicle_queue.popleft()
        #get the current vehicle's depart time
        current_depart = current_vehicle.get_depart()
        #get the current vehicle's duration
        current_duration = current_vehicle.get_duration()
        #get the current vehicle's arrival time
        current_arrival = current_vehicle.get_arrival()
        #get the current vehicle's train time
        current_train = current_vehicle.get_train()
        #get the current vehicle's communication time
        current_communication = current_vehicle.get_communication()
        #get the current vehicle's alpha
        current_alpha = current_vehicle.get_alpha()
        #update the current vehicle's total time
        current_total_time = current_train + current_communication  
        return current_vehicle, current_depart, current_duration, current_arrival, current_train, current_communication, current_alpha, current_total_time
    
    def step_1(self,action):
        current_vehicle, current_depart, current_duration, current_arrival, current_train, current_communication, current_alpha, current_total_time = self.get_current_vehicle()
        #update the current time of environment
        self.time = current_depart    
        #check if the current vehicle can be scheduled
        if (current_train/current_alpha) + current_communication > current_duration:
            return self.get_state(), 0, False, {"vehicle %d cannot be scheduled, because it cannot compelte train and communication in duration time.", current_vehicle.get_id()}
        elif current_total_time > current_duration:
            current_train = current_duration - current_communication  
        else:
            current_train = current_train  
        #set the selected resource block as a list with three elements, the first element is the start time of the resource block, the second element is the end time of the resource block, and the third element is the bandwith index of the resource block    
        selected_resource_block = [current_depart+current_train, current_depart+current_train+current_communication, 0]
        done = False
        resource_block = self.resource_block_decoder(self.current_state[action])
        #get the start time of the resource block
        rb_start_time = resource_block[0]
        #get the end time of the resource block
        rb_end_time = resource_block[1]
        #get the bandwidth of the resource block
        rb_bandwidth = resource_block[2]
        #get the duration of the resource block
        candidate_duration = rb_end_time - rb_start_time
        #check if the resource block is available
        if candidate_duration >= current_communication:
            #check if the resource block is available, if the resource block is available, update the selected resource block
            if selected_resource_block[0] >= rb_start_time and selected_resource_block[1] <= rb_end_time:
                selected_resource_block[2] = rb_bandwidth
            elif selected_resource_block[0] > rb_start_time and selected_resource_block[1] > rb_end_time:
                gap = selected_resource_block[1] - rb_end_time
                required_ratio = current_vehicle.get_train()/(current_train - gap)
                if required_ratio <= current_alpha:
                    current_train = current_train - gap
                    selected_resource_block[0] = resource_block[0] - gap
                    selected_resource_block[1] = resource_block[1] - gap
                    selected_resource_block[2] = rb_bandwidth

        if selected_resource_block[2] == 0:
            return self.get_state(), 0, False, {"vehicle %d cannot be scheduled, because there is no available resource block.", current_vehicle.get_id()}
        #get the corresponding resource pool
        selected_resource_pool = self.resourcepool[str(selected_resource_block[2])]
        #obtain the available resource block list of the corresponding resource pool which is same as the selected resource block start time, update the start time of the available resource block
        resource_pool_update_flag = False
        for available_resource_block in selected_resource_pool:
            if available_resource_block[0]<selected_resource_block[0] and available_resource_block[1]>selected_resource_block[0]:
                divide_resource_block = [0,available_resource_block[1]]
                available_resource_block[1] = round(selected_resource_block[0])
                divide_resource_block[0] = round(selected_resource_block[1])
                selected_resource_pool.append(divide_resource_block)
                resource_pool_update_flag = True
                break
            elif available_resource_block[0] == selected_resource_block[0] and available_resource_block[1] == selected_resource_block[1]:
                #delete the selected resource block from the available resource block list
                selected_resource_pool.remove(available_resource_block)
                resource_pool_update_flag = True
                break
            elif available_resource_block[0] < selected_resource_block[0] and available_resource_block[1] == selected_resource_block[1]:
                available_resource_block[1] = round(selected_resource_block[0])
                resource_pool_update_flag = True
                break
            elif available_resource_block[0] == selected_resource_block[0] and available_resource_block[1] > selected_resource_block[1]:
                available_resource_block[0] = round(selected_resource_block[1])
                resource_pool_update_flag = True
                break
            else:
                resource_pool_update_flag = False
                continue
        if resource_pool_update_flag == False:
            return self.get_state(), 0, False, {"Fail to update the resource pool by RB [%d,%d,%d]", selected_resource_block[0], selected_resource_block[1], selected_resource_block[2]}
        
        
        #update the scheduled vehicle list
        self.scheduled_vehicle.append(current_vehicle)
        #return the next state, reward, done, and info
        if len(self.vehicle_queue) == 0:
            print("The vehicle list is empty, the environment is done.")
            next_state = self.get_state()
        else:
            next_state = self.get_state()
        reward = self.get_reward(selected_resource_block)
        info = {"Success to schedule the vehicle %d", current_vehicle.get_id()}
        
        return next_state, reward, done, info
        
        
    
    def step_2(self, action):
        current_vehicle, current_depart, current_duration, current_arrival, current_train, current_communication, current_alpha, current_total_time = self.get_current_vehicle()
        #update the current time of environment
        self.time = current_depart    
        #check if the current vehicle can be scheduled
        if (current_train/current_alpha) + current_communication > current_duration:
            selected_action = action[0]
            return self.get_state(), 0, False, {"vehicle %d cannot be scheduled, because it cannot compelte train and communication in duration time.", current_vehicle.get_id()}, selected_action
        elif current_total_time > current_duration:
            current_train = current_duration - current_communication  
        else:
            current_train = current_train  
        #set the selected resource block as a list with three elements, the first element is the start time of the resource block, the second element is the end time of the resource block, and the third element is the bandwith index of the resource block    
        selected_resource_block = [current_depart+current_train, current_depart+current_train+current_communication, 0]
        '''
        The above parts are the details of the current vehicle, and the following parts are the details of the current action
        The current action is a list of available resources blocks, and the each element of the list is a list of two elements, the first element is the start time of the resource block, and the second element is the end time of the resource block
        For example, if the current action is [[0, 1000], [1000, 2000]], it means that the current vehicle can use the resource block from 0 to 1000 or from 1000 to 2000 and the third in the element of the list is the bandwith index of the resource block
        The action list is sorted by the output of the model, and the first element of the list is the best choice
        '''
        done = False
        action = np.delete(action, np.where(action == 0))
        selected_action = 0
        for encode_resource_block_index in action:
            encode_resource_block = self.current_state[encode_resource_block_index]
            if encode_resource_block == 0:
                continue  
            resource_block = self.resource_block_decoder(encode_resource_block)
            #get the start time of the resource block
            rb_start_time = resource_block[0]
            #get the end time of the resource block
            rb_end_time = resource_block[1]
            #get the bandwidth of the resource block
            rb_bandwidth = resource_block[2]
            #get the duration of the resource block
            candidate_duration = rb_end_time - rb_start_time
            #check if the resource block is available
            if candidate_duration >= current_communication:
                #check if the resource block is available, if the resource block is available, update the selected resource block
                if selected_resource_block[0] >= rb_start_time and selected_resource_block[1] <= rb_end_time:
                    selected_resource_block[2] = rb_bandwidth
                    selected_action = encode_resource_block_index
                    break
                elif selected_resource_block[0] > rb_start_time and selected_resource_block[1] > rb_end_time:
                    gap = selected_resource_block[1] - rb_end_time
                    required_ratio = current_vehicle.get_train()/(current_train - gap)
                    if required_ratio <= current_alpha:
                        current_train = current_train - gap
                        selected_resource_block[0] = resource_block[0] - gap
                        selected_resource_block[1] = resource_block[1] - gap
                        selected_resource_block[2] = rb_bandwidth
                        selected_action = encode_resource_block_index
                        break
                else:
                    continue

        
        ''''
        The following step is because the action just contains the resource block of total time, but the vehicle needs a part of the resource block, so we need to update the end time of selected resource block.
        '''
        if selected_resource_block[2] == 0:
            selected_action = action[0]
            return self.get_state(), 0, False, {"vehicle %d cannot be scheduled, because there is no available resource block.", current_vehicle.get_id()}, selected_action
        #get the corresponding resource pool
        selected_resource_pool = self.resourcepool[str(selected_resource_block[2])]
        #obtain the available resource block list of the corresponding resource pool which is same as the selected resource block start time, update the start time of the available resource block
        resource_pool_update_flag = False
        for available_resource_block in selected_resource_pool:
            if available_resource_block[0]<selected_resource_block[0] and available_resource_block[1]>selected_resource_block[0]:
                divide_resource_block = [0,available_resource_block[1]]
                available_resource_block[1] = round(selected_resource_block[0])
                divide_resource_block[0] = round(selected_resource_block[1])
                selected_resource_pool.append(divide_resource_block)
                resource_pool_update_flag = True
                break
            elif available_resource_block[0] == selected_resource_block[0] and available_resource_block[1] == selected_resource_block[1]:
                #delete the selected resource block from the available resource block list
                selected_resource_pool.remove(available_resource_block)
                resource_pool_update_flag = True
                break
            elif available_resource_block[0] < selected_resource_block[0] and available_resource_block[1] == selected_resource_block[1]:
                available_resource_block[1] = round(selected_resource_block[0])
                resource_pool_update_flag = True
                break
            elif available_resource_block[0] == selected_resource_block[0] and available_resource_block[1] > selected_resource_block[1]:
                available_resource_block[0] = round(selected_resource_block[1])
                resource_pool_update_flag = True
                break
            else:
                resource_pool_update_flag = False
                continue
        if resource_pool_update_flag == False:
            return self.get_state(), 0, False, {"Fail to update the resource pool by RB [%d,%d,%d]", selected_resource_block[0], selected_resource_block[1], selected_resource_block[2]}
        
        
        #update the scheduled vehicle list
        self.scheduled_vehicle.append(current_vehicle)
        #return the next state, reward, done, and info
        if len(self.vehicle_queue) == 0:
            print("The vehicle list is empty, the environment is done.")
            next_state = self.get_state()
        else:
            next_state = self.get_state()
        reward = self.get_reward(selected_resource_block)
        info = {"Success to schedule the vehicle %d", current_vehicle.get_id()}
        
        return next_state, reward, done, info, selected_action
     
    def reset(self):
        self.time = 0
        self.vehicle_queue = deque(self.vehicle_list)
        self.scheduled_vehicle = []
        for i in range(1, AVAILABLE_BANDWIDTH+1):
            self.resourcepool[str(i)] = [[0, ONE_ROUND_TIME]]
        return self.get_state()
    
    def get_state(self):
        current_state = []
        if len(self.vehicle_queue) == 0:
            vehicle = Vehicle(0, 0, 0, 0, 0, 0, 0)
        else:    
            vehicle = self.vehicle_queue[0]
        depart = vehicle.get_depart()
        arrival = vehicle.get_arrival()
        communication = vehicle.get_communication()
        resourcepool = []
        for i in range(1, AVAILABLE_BANDWIDTH+1):
            for resource_block in self.resourcepool[str(i)]:
                duration = resource_block[1] - resource_block[0]
                #sort the resource block by the duration of the resource block
                if duration >= communication and resource_block[0] < arrival and resource_block[1] > depart:
                    coded_resource_block = self.resource_block_encoder(resource_block,i)
                    decode_resource_block = self.resource_block_decoder(coded_resource_block)
                    resourcepool.append(decode_resource_block)
        #sort the resourcepool by the dureation of the resource block
        resourcepool = sorted(resourcepool, key = lambda x: x[1]-x[0], reverse = True)
        if len(resourcepool) < self.action_dim:
            for i in range(self.action_dim - len(resourcepool)):
                resourcepool.append([0,0,0])
        else:
            resourcepool = resourcepool[:self.action_dim]
        
        for resource_block in resourcepool:
            current_state.append(self.resource_block_encoder(resource_block, resource_block[2]))
        
        current_state.append(depart)
        current_state.append(arrival)
        current_state.append(communication)
        self.current_state = current_state
        return current_state
        
        
        
    def get_state_old(self):
        current_state = []
        #get the current vehicle atricbutes
        if len(self.vehicle_queue) == 0:
            vehicle = Vehicle(0, 0, 0, 0, 0, 0, 0)
        else:    
            vehicle = self.vehicle_queue[0]
        depart = vehicle.get_depart()
        arrival = vehicle.get_arrival()
        communication = vehicle.get_communication()
        #get the first action_dim-th available resource block list of the corresponding resource pool
        resourcepool = []
        '''
        Here the resource pool is a list with 2D elements, we need to an encoder to encode the resource block, the encoder is as follows:
        because the ONE_ROUND_TIME is 1000, so the interval of the resource block is [0,1000] of bandwidth 3, the encoder will encode the resource block as follows: 1 0000 1000.
        Therefore, we use a 9-digit number to encode the resource block, the first digit is the bandwith index, the second to the fourth digit is the start time of the resource block, and the fifth to the ninth digit is the end time of the resource block.
        '''
        for i in range(1, AVAILABLE_BANDWIDTH+1):
            for resource_block in self.resourcepool[str(i)]:
                duration = resource_block[1] - resource_block[0]
                basecode = i*100000000
                if duration >= communication:
                    #encode the resource block by add it bandwith index * 10000
                    coded_resource_block = self.resource_block_encoder(resource_block,i)
                    resourcepool.append(coded_resource_block)
                else:
                    resourcepool.append(000000000)
        #add the current vehicle's depart time, arrival time, and communication time to the state
        current_state = np.array(resourcepool)
        #extend the current state to the state_dim and use the basecode 000000000 to fill the rest of the state
        while len(current_state) < self.state_dim:
            current_state = np.append(current_state, 000000000)
        #update the current state with the current vehicle's depart time, arrival time, and communication on the final three elements of the state
        current_state[-3] = depart
        current_state[-2] = arrival
        current_state[-1] = communication
        self.current_state = current_state
        return current_state
    
    def get_reward(self, selected_resource_block):
        duration = selected_resource_block[1] - selected_resource_block[0]
        scheduled_vehicle_count = len(self.scheduled_vehicle)
        improved_resource_utilization = duration/(ONE_ROUND_TIME*AVAILABLE_BANDWIDTH)
        reward = 1 + improved_resource_utilization
        return reward   
    
    def resource_block_encoder(self, resource_block, bandwith_index):
        basecode = bandwith_index*100000000
        coded_resource_block = basecode + resource_block[0] + resource_block[1]*10000
        return coded_resource_block
    
    def resource_block_decoder(self, coded_resource_block):
        bandwith_index = int(coded_resource_block/100000000)
        temp = coded_resource_block%100000000
        start_time = int(temp%1000000)
        end_time = round(temp/10000)
        return [start_time, end_time, bandwith_index]