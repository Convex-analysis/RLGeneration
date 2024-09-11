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


lowwithoutfilter = [792.34,201,11.3]#average watting time=213.33,surplus=1

lowwithoutfiltersurplus = [993.658,415,23]#average watting time=232,surplus=2

lowwithfilter = [781.3,580,20]#average watting time=118

lowDSR_results = [619.88,533,19.993]#no waiting time,surplus=1

def high_density_time_bar_plot():
    labels = ['Without \n Filter', 'Without Filter \n (Surplus = 2)', 'With Filter', 'DSR']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [highwithoutfilter[0], highwithoutfiltersurplus[0], highwithfilter[0], highDSR_results[0]], width, label='Average Round Wall Time')
    rects2 = ax.bar(x + width/2, [highwithoutfilter[1], highwithoutfiltersurplus[1], highwithfilter[1], highDSR_results[1]], width, label='Longest Schedule Vehicle')
    ax.set_ylabel('Time')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

def high_density_number_bar_plot():
    labels = ['Without \n Filter', 'Without Filter \n (Surplus = 2)', 'With Filter', 'DSR']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width/2, [highwithoutfilter[2], highwithoutfiltersurplus[2], highwithfilter[2], highDSR_results[2]], width, label='Scheduled Number')
    ax.set_ylabel('Number')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.show()

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
    #plot(path + filename, "episode", "reward")
    high_density_number_bar_plot()