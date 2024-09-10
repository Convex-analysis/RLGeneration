import matplotlib 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

#[current_round_period, longest_schedule_vhicle, scheduled_number]
#from highdensity\constantCSscheme_09-10-11-37.csv
highwithoutfilter = [792.34,201,11.3]#average watting time=213.33,surplus=1
#from highdensity\constantCSscheme_09-10-14-27.csv
highwithoutfiltersurplus = [993.658,415,23]#average watting time=232,surplus=2
#from highdensity\constantCSscheme_09-10-14-37.csv
highwithfilter = [781.3,580,20]#average watting time=118
#from highdensity\highresult.csv
highDSR_results = [619.88,533,19.993]#no waiting time,surplus=1


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
    filename = "ACER_resource_allocation_log_9.csv"
    plot(path + filename, "episode", "reward")