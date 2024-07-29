import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Plotting function from a csv file to a 2d plot
def plot(filename,x, y):
    #load data from csv file as a dataframe
    data = pd.read_csv(filename)
    #extract the column "episode" as x-axis and "reward" as y-axis
    x = data[x]
    y = data[y]
    #plot the data
    plt.plot(x, y)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward vs Episode')
    plt.show()

if __name__ == "__main__":
    path = "ACER_files/resource_allocation/stability/"
    filename = "ACER_resource_allocation_log_8.csv"
    plot(path + filename, "episode", "reward")