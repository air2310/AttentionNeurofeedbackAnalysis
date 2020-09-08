import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

mean_subsweek = 5
max_subsweek = 8

min_substotal = 60
max_substotal = 80

subs_collected = 31 + 28

min_subsneeded = min_substotal - subs_collected
max_subsneeded = max_substotal - subs_collected
mean_subsneeded = (max_subsneeded+min_subsneeded)/2
range_subsneeded = max_subsneeded-min_subsneeded
SD_subsneeded = range_subsneeded / 10

n_simulations = 1000
Weeksall = np.empty((n_simulations))

subcount_all = np.round(np.random.randn(n_simulations)*SD_subsneeded + mean_subsneeded)
# subcount_all = np.round(np.random.rand(n_simulations)*range_subsneeded + min_subsneeded) # uniform dist

# simulate

for simulation in np.arange(n_simulations):
    collecting, week, subcount = True, 0, 0
    while (collecting==True):
        # add a week of data collection
        week +=1

        # simulate collecting subjects
        tmp = np.round(np.random.randn(1) + mean_subsweek)
        if (tmp < 0): tmp = 0
        if (tmp > max_subsweek): tmp = max_subsweek
        subcount += tmp

        if (subcount >= subcount_all[simulation]):
            Weeksall[simulation] = week
            collecting = False


fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
ax1.hist(Weeksall, bins = len(np.unique(Weeksall)))
ax1.set_xlabel('Number of weeks needed')
ax1.set_title('Weeks to go estimates')

# plot number of subs needed hist
ax2.hist(subcount_all + subs_collected, bins = len(np.unique(subcount_all)))
ax2.set_xlabel('number of participants')
ax2.set_title('Total participants estimate')

tmp = np.round(np.random.randn(1000) + mean_subsweek)
tmp[tmp < 0] = 0
tmp[tmp > max_subsweek] = max_subsweek
ax3.hist(tmp, bins = len(np.unique(tmp)))
ax3.set_xlabel('number of participants/ week')
ax3.set_title('weekly participant count')

direct_resultsroot = Path("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/")
plt.savefig(direct_resultsroot / Path("datacollectiontimeline" + '.png'), format='png')