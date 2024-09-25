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

#from constantCSscheme_09-11-14-04.csv
lowwithoutfilter = [832.4,410,11.6]#average watting time=124.7,surplus=1
#from  constantCSscheme_09-11-14-23.csv
lowwithoutfiltersurplus = [998,824,18.8]#average watting time=126.5,surplus=2
#from  constantCSscheme_09-11-14-26.csv
lowwithfilter = [906.36,888.78,18.1]#average watting time=42.37
#from  result.csv
lowDSR_results = [808.12,702.46,19.1]#no waiting time,surplus=1


#all selected vehicles
allhighwithoutfilter = [998.2,841,32.5]#average watting time=211.3
allhighwithoutfiltersurplus = [998.2,841,32.5]#average watting time=211.3
allhighwithfilter = [994,934,37.4]#average watting time=136.2
allhighDSR_results = [929.4,815,42.9]#no waiting time,surplus=1

alllowwithoutfilter = [998.16,824,18.8]#average watting time=126.54
alllowwithoutfiltersurplus = [998.2,824,18.8]#average watting time=126.54
alllowwithfilter = [926,914.88,18.8]#average watting time=41.66
alllowDSR_results = [965.9,799.7,24.8]#no waiting time,surplus=1

def high_density_time_bar_plot():
    labels = ['FedAvg', 'FEDDATE-CS \n (Surplus = 2)', 'AVFL', 'MR-VFL \n Scheduler']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.axhline(y=1000, color='gray', linestyle='--')
    rects1 = ax.bar(x - width/2, [highwithoutfilter[0], highwithoutfiltersurplus[0], highwithfilter[0], highDSR_results[0]], width, label='Average Round Wall Time')
    rects2 = ax.bar(x + width/2, [highwithoutfilter[1], highwithoutfiltersurplus[1], highwithfilter[1], highDSR_results[1]], width, label='Longest Schedule Vehicle')
    ax.set_ylabel('Time')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('high_density_time_bar_plot.png')
    plt.show()
    

def low_density_time_bar_plot():
    labels = ['FedAvg', 'FEDDATE-CS \n (Surplus = 2)', 'AVFL', 'MR-VFL \n Scheduler']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.axhline(y=1000, color='gray', linestyle='--')
    rects1 = ax.bar(x - width/2, [lowwithoutfilter[0], lowwithoutfiltersurplus[0], lowwithfilter[0], lowDSR_results[0]], width, label='Average Round Wall Time')
    rects2 = ax.bar(x + width/2, [lowwithoutfilter[1], lowwithoutfiltersurplus[1], lowwithfilter[1], lowDSR_results[1]], width, label='Longest Schedule Vehicle')
    ax.set_ylabel('Time')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('low_density_time_bar_plot.png')
    plt.show()
    


def scheduler_number_bar_plot():
    labels = ['FedAvg', 'FEDDATE-CS \n (Surplus = 2)', 'AVFL', 'MR-VFL \n Scheduler']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    # add a line at y-axis = 20
    ax.axhline(y=20, color='gray', linestyle='--')
    rects1 = ax.bar(x - width/2, [highwithoutfilter[2], highwithoutfiltersurplus[2], highwithfilter[2], highDSR_results[2]], width, label='High Density Scenario')
    rects2 = ax.bar(x + width/2, [lowwithoutfilter[2], lowwithoutfiltersurplus[2], lowwithfilter[2], lowDSR_results[2]], width, label='Low Density Scenario')
    ax.set_ylabel('Scheduled Number')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('scheduler_number_bar_plot.png')
    plt.show()



def greedy_high_density_time_bar_plot():
    labels = ['FedAvg', 'FEDDATE-CS \n (Surplus = 2)', 'AVFL', 'MR-VFL \n Scheduler']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.axhline(y=1000, color='gray', linestyle='--')
    rects1 = ax.bar(x - width/2, [allhighwithoutfilter[0], allhighwithoutfiltersurplus[0], allhighwithfilter[0], allhighDSR_results[0]], width, label='Average Round Wall Time')
    rects2 = ax.bar(x + width/2, [allhighwithoutfilter[1], allhighwithoutfiltersurplus[1], allhighwithfilter[1], allhighDSR_results[1]], width, label='Longest Schedule Vehicle')
    ax.set_ylabel('Time')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('greedy_high_density_time_bar_plot.png')
    plt.show()
    

def greedy_low_density_time_bar_plot():
    labels = ['FedAvg', 'FEDDATE-CS \n (Surplus = 2)', 'AVFL', 'MR-VFL \n Scheduler']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    ax.axhline(y=1000, color='gray', linestyle='--')
    rects1 = ax.bar(x - width/2, [alllowwithoutfilter[0], alllowwithoutfiltersurplus[0], alllowwithfilter[0], alllowDSR_results[0]], width, label='Average Round Wall Time')
    rects2 = ax.bar(x + width/2, [alllowwithoutfilter[1], alllowwithoutfiltersurplus[1], alllowwithfilter[1], alllowDSR_results[1]], width, label='Longest Schedule Vehicle')
    ax.set_ylabel('Time')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('low_density_time_bar_plot.png')
    plt.show()


def greedy_scheduler_number_bar_plot():
    labels = ['FedAvg', 'FEDDATE-CS \n (Surplus = 2)', 'AVFL', 'MR-VFL \n Scheduler']
    x = np.arange(len(labels))
    width = 0.35
    fig, ax = plt.subplots()
    # add a line at y-axis = 20
    #ax.axhline(y=20, color='gray', linestyle='--')
    rects1 = ax.bar(x - width/2, [allhighwithoutfilter[2], allhighwithoutfiltersurplus[2], allhighwithfilter[2], allhighDSR_results[2]], width, label='High Density Scenario')
    rects2 = ax.bar(x + width/2, [alllowwithoutfilter[2], alllowwithoutfiltersurplus[2], alllowwithfilter[2], alllowDSR_results[2]], width, label='Low Density Scenario')
    ax.set_ylabel('Scheduled Number')
    ax.set_title('Comparison of Different Scheduling Strategies')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    plt.savefig('greedy_scheduler_number_bar_plot.png')
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
    scheduler_number_bar_plot()
    high_density_time_bar_plot()
    low_density_time_bar_plot()
    greedy_scheduler_number_bar_plot()
    greedy_high_density_time_bar_plot()
    greedy_low_density_time_bar_plot()