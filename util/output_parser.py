from xml.dom.minidom import parse
import xml.dom.minidom
import pandas as pd
import numpy as np
import math
import os
from env import Vehicle

def load_csv(file):
    df = pd.read_csv(file)
    df = df.sort_values(by = "depart")
    vehicle_list = []
    for index, row in df.iterrows():
        vehicle = Vehicle(row["id"], row["depart"], row["duration"], row["arrival"], row["train"], row["communication"], row["alpha"],row["select_times"],row["data_quality"])
        vehicle_list.append(vehicle)
    return vehicle_list


def xml_parser(path, filename):
    filepath = path + filename
    Domtree = xml.dom.minidom.parse(filepath)
    collection = Domtree.documentElement
    trips = collection.getElementsByTagName("tripinfo")
    #initialize the 4d numpy array
    nparray = np.array([["id", "depart", "duration", "arrival","train", "communication", "alpha","select_times","data_quality"]])
    #given the required epoch number
    epochs = 10
    #set the random seed
    np.random.seed(0)
    #iterate through all the nodes in the xml file
    for node in trips:
        results = generate_other_simulated_attributes(node, epochs)
        nparray = np.append(nparray, [results], axis = 0)
        
        #wirte the output to a csv file
        
    #delete the first row of the array
    dfcolumns = nparray[0]
    nparray = np.delete(nparray, 0, 0)    
    df = pd.DataFrame(nparray, columns = dfcolumns)
    #sort the dataframe by the depart time
    df = df.sort_values(by = "depart")
    csv_name = filename.split(".")[0] + ".csv"
    #check if the csv file exists
    if not (csv_name in os.listdir(path)):    
        df.to_csv(path + csv_name, index = False)


def generate_other_simulated_attributes(node, required_epochs):
    v_id = node.getAttribute("id")
    depart = node.getAttribute("depart")
    duration = node.getAttribute("duration")
    arrival = node.getAttribute("arrival")
        
    #transform the string to float
    duration = float(duration)
    #print("duration: ", duration)
    #let the sum of communication time and computation time be randomly generated from 0.5 to 1.5 times run_time
    total_time =  np.random.randint(duration *0.5, duration *1.5)

     #the ratio of communication time to computation time is randomly generated from 0.5 to 0.9
     #ratio = np.random.uniform(0.6, 0.9)
    communication_time = np.random.normal(30, 5)
    train_time = np.random.normal(180, 20)
    #generate the least epoch number accoding to the given epochs and a guassian distribution
    least_epoch = np.random.normal(required_epochs*0.1, 1)
    if least_epoch <= 1:
        least_epoch = 1
    elif least_epoch > required_epochs:
        least_epoch = required_epochs
    else:
        least_epoch = round(least_epoch)
     #generate the upper bound of alpha according to the at least epoch number
    alpha = int(required_epochs/least_epoch)
    
    #generate the data quality
    """
    data_quality is affected by two factors: data categories and data amounts, both of them quantified by integers according to the following rules:
    refer to the MNIST dataset, and cifar10 dataset, total size of the dataset is 60000
    data categories: assume total 10 categories and random sample from interval [6,10] 
    Assuem maximum data size is 3000, and minimum data size is 600, and random sample from interval [600, 6000]
    data amounts: 
    """
    full_size, full_categories = 60000, 10
    data_categories = np.random.randint(6, 11)
    data_amounts = np.random.randint(600, 6000)
    data_quality = 0.6 * data_categories/full_categories + 0.4 * data_amounts/full_size
    
    #generate the select times
    select_times = np.random.randint(1, 5)

    return [v_id, depart, duration, arrival, train_time, communication_time, alpha, select_times, data_quality]

        
        


        
        