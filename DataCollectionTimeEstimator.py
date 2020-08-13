import numpy as np
import matplotlib.pyplot as plt


mean_subsweek = 5
max_subsweek = 8

min_substotal = 40
max_substotal = 80

subs_collected = 22

min_subsneeded = min_substotal - subs_collected
max_subsneeded = max_substotal - subs_collected
mean_subsneeded = (max_subsneeded+min_subsneeded)/2
range_subsneeded = max_subsneeded-min_subsneeded
SD_subsneeded = range_subsneeded / 6

n_simulations = 1000
Weeksall = np.empty((n_simulations))

subcount_all = np.round(np.random.randn(n_simulations)*SD_subsneeded + mean_subsneeded)
# subcount_all = np.round(np.random.rand(n_simulations)*range_subsneeded + min_subsneeded) # uniform dist

# plot number of subs needed hist
plt.figure()
plt.hist(subcount_all, bins = len(np.unique(subcount_all)))

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


plt.figure()
plt.hist(Weeksall, bins = len(np.unique(Weeksall)))
plt.xlabel('Number of weeks needed')
plt.ylabel('Probability')