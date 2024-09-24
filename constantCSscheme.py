import os
import time
import math
import pandas as pd
import numpy as np
from collections import deque
from env import Vehicle
from util.output_parser import load_csv, xml_parser

# Constants for the one round time of FL corresponding to the simulation time of SUMO
ONE_ROUND_TIME = 1000
REQUIRED_EPOCHS = 10
AVAILABLE_BANDWIDTH = 3
REQUIRED_CLIENTS = 100

class ConstantCSscheme:
    def __init__(self):
        self.vehicle_list = []
        self.atLeastClients = REQUIRED_CLIENTS
        self.completedTimeList = []
        ResourcePool = {}
        self.resourcepool = ResourcePool
    
    def initialize_resource_pool(self):
        ResourcePool = {}
        for i in range(1, AVAILABLE_BANDWIDTH+1):
            ResourcePool[str(i)] = [[0, ONE_ROUND_TIME]]
        self.resourcepool = ResourcePool
    
    def select_vehicle(self, if_filter, vehilce_list,surplus = 1):
        #initialize the vehicle list
        self.initialize_resource_pool()
        self.vehicle_list = []
        if if_filter:
            return self.check_clients_number_with_filter(vehilce_list)
        else:
            return self.check_clients_number_without_filter(vehilce_list,surplus)
    #don't consider the filter but make surplus vehicles 
    def check_clients_number_without_filter(self,vehilce_list, surplus):
        current_clients = vehilce_list
        longest_depart_time = 0
        success_number = 0
        for vehicle in current_clients:
            dwell_time = vehicle.get_duration()
            enteire_time = vehicle.get_communication() + vehicle.get_train()
            completed_time = vehicle.get_depart()
            success_number += 1
            self.vehicle_list.append(vehicle)
            if completed_time > longest_depart_time:
                longest_depart_time = completed_time
            if success_number > (self.atLeastClients * surplus - 1):
                return longest_depart_time, success_number
        return longest_depart_time, success_number
            
    def check_clients_number_with_filter(self,vehilce_list):
        current_clients = vehilce_list
        longest_time = 0
        success_number = 0
        for vehicle in current_clients:
            dwell_time = vehicle.get_duration()
            enteire_time = vehicle.get_communication() + vehicle.get_train()
            completed_time = vehicle.get_depart() + enteire_time
            if enteire_time <= dwell_time:
                success_number += 1
                self.vehicle_list.append(vehicle)
                if completed_time > longest_time:
                    longest_time = completed_time
            if success_number > self.atLeastClients-1:
                return longest_time, success_number
        return longest_time, success_number
        
    def find_longest_time(self):
        longest_time = 0
        for channel in self.resourcepool.keys():
            for rb in self.resourcepool[channel]:
                if rb[0] > longest_time:
                    longest_time = rb[0]
        return longest_time
        
    def calculate_clients_waiting_time(self):
        canadinate_vehicle_list = self.get_vehicle_list()
        waiting_time_list = []
        scheduled_vehicle_number = 0
        for vehicle in canadinate_vehicle_list:
            current_schedule_flag = False
            communication_time_start = vehicle.get_depart() + vehicle.get_train()
            communication_time_end = communication_time_start + vehicle.get_communication()
            for channel in self.resourcepool.keys():
                for idx,rb in enumerate(self.resourcepool[channel]):
                    rb_inteval = rb[1] - rb[0]
                    if rb_inteval <= 0:
                        continue
                    # if the communication time is within the resource block
                    if communication_time_start >= rb[0] and communication_time_end <= rb[1]:
                        self.update_resource_pool(channel, idx, communication_time_start, communication_time_end)
                        current_schedule_flag = True
                        break
                    # if the required communication time is less than the resource block and the beginning of the resource block is later than the communication time, thus the vehicle can wait for the resource block
                    elif rb_inteval >= vehicle.get_communication() and rb[0] > communication_time_start:
                        waiting_time = rb[0] - communication_time_start
                        communication_time_start = rb[0]
                        communication_time_end = communication_time_end + waiting_time
                        self.update_resource_pool(channel, idx, communication_time_start, communication_time_end)
                        #record the waiting time
                        waiting_time_list.append(waiting_time)
                        current_schedule_flag = True
                        break 
                    else:
                        continue
                if current_schedule_flag:
                    if vehicle.get_communication() + vehicle.get_train() <= vehicle.get_duration():
                        scheduled_vehicle_number += 1
                    break
        peroid_time = self.find_longest_time()                                      
        return waiting_time_list, scheduled_vehicle_number,peroid_time
    
    def update_resource_pool(self, channel, rb_idx, new_rb_start, new_rb_end):
        old_rb_start = self.resourcepool[channel][rb_idx][0]
        old_rb_end = self.resourcepool[channel][rb_idx][1]
        temp_rb = [old_rb_start, new_rb_start]
        temp_rb2 = [new_rb_end, old_rb_end]
        self.resourcepool[channel].remove(self.resourcepool[channel][rb_idx])
        if temp_rb[1] - temp_rb[0] > 0:
            self.resourcepool[channel].append(temp_rb)
        if temp_rb2[1] - temp_rb2[0] > 0:
            self.resourcepool[channel].append(temp_rb2)
        self.resourcepool[channel].sort()
        return
        
    def get_vehicle_list(self):
         return self.vehicle_list
     
    def set_vehicle_list(self, vehicle_list):
         self.vehicle_list = vehicle_list
         

if __name__ == "__main__":
    cc = ConstantCSscheme()
    round_idx = 1
    #defalut_path = "./highdensity/"
    defalut_path = "./lowdensity/"
    dataframe = pd.DataFrame()
    #control the number of surplus vehicles on the without filter function
    surplus = 2
    # the first row of the dataframe is the column name, which are round, required_clients, scheduled_clients, average_waiting_time, wall_time
    dataframe = pd.DataFrame(columns = ["round", "longest_depart_time", "scheduled_clients", "average_waiting_time", "wall_time", "surplus"])
    # load the vehicle list from all .xml file
    for file in os.listdir(defalut_path):
        if file.endswith(".xml") and file.startswith("Trip"):
            print("Loading file: ", file)
            xml_parser(defalut_path, file)
            file = file.split(".")[0] + ".csv"
            vehicle_list = load_csv(defalut_path + file)
            longesttime, selected_number = cc.select_vehicle(True, vehicle_list, surplus)
            waiting_time_list, scheduled_number, peroid = cc.calculate_clients_waiting_time()
            print("The longest time is: ", longesttime)
            print("The number of vehicles that can be scheduled is: ", selected_number)
            print("The number of vehicles that have been scheduled is: ", scheduled_number)
            if len(waiting_time_list) == 0:
                print("No vehicle is waiting for the resource block")
            else:
                print("The average waiting time is: ", np.mean(waiting_time_list))
            print("The wall time of thie round is: ", peroid)
            dataframe.loc[round_idx] = [round_idx, longesttime, scheduled_number, np.mean(waiting_time_list), peroid,surplus]
            round_idx += 1
    #obtain current time to name the result file

    current_time = time.strftime("%m-%d-%H-%M", time.localtime())
    result_file = defalut_path + "constantCSscheme_" + current_time + ".csv"
    dataframe.to_csv(result_file, index = False)
            
            
    