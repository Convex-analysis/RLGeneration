from xml.dom.minidom import parse
import xml.dom.minidom
import pandas as pd
import numpy as np

Domtree = xml.dom.minidom.parse("output_file.xml")
collection = Domtree.documentElement
trips = collection.getElementsByTagName("tripinfo")
#initialize the 4d numpy array
nparray = np.array([["id", "depart", "duration", "arrival","train", "communication", "alpha"]])
#given the required epoch number
epochs = 10
#set the random seed
np.random.seed(0)
#iterate through all the nodes in the xml file
for node in trips:
    #get the values of the attributes
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
    communication_time = np.random.normal(20, 5)
    train_time = np.random.normal(120, 20)
    #generate the least epoch number accoding to the given epochs and a guassian distribution
    least_epoch = np.random.normal(epochs*0.5, 3)
    if least_epoch <= 1:
        least_epoch = 1
    elif least_epoch > epochs:
        least_epoch = epochs
    else:
        least_epoch = round(least_epoch)
    print("least_epoch: ", least_epoch)
    alpha = int(epochs/least_epoch)
    nparray = np.append(nparray, [[v_id, depart, duration, arrival,train_time, communication_time, alpha]], axis = 0)
    
    #wirte the output to a csv file
    
#delete the first row of the array
dfcolumns = nparray[0]
nparray = np.delete(nparray, 0, 0)    
df = pd.DataFrame(nparray, columns = dfcolumns)
#sort the dataframe by the depart time
df = df.sort_values(by = "depart")    
df.to_csv("vehicle_settings.csv", index = False)


    
    


    
    