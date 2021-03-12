# Import nescescary packages
import numpy as np
from pathlib import Path
# import mne
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
# import pandas as pd
# import sys

import helperfunctions_ATTNNF as helper


def run(settings, sub_val):
    settings = settings.get_settings_behave_duringNF()

    # preallocate
    frames_trial = settings.trialduration * settings.mon_ref
    feedback = np.empty((settings.num_trials, frames_trial, settings.num_days))
    predictedstate = np.empty((settings.num_trials, frames_trial, settings.num_days))
    moveonsets = np.empty((settings.num_movements, settings.num_trials, settings.num_days))

    # loop through days
    for day_count, day_val in enumerate(settings.daysuse):
        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)

        # decide which file to use
        possiblefiles = []
        filesizes = []
        for filesfound in bids.direct_data_behave.glob(bids.filename_behave + "*.mat"):
            filesizes.append(filesfound.stat().st_size)
            possiblefiles.append(filesfound)

        if any(filesizes):
            file2useIDX = np.argmax(filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
            file2use = possiblefiles[file2useIDX]
        else:
            if (day_count==0):
                 df_behavedata=[] # assign df_behave data as empty if this happens on the first day
            continue

        # load data
        F = h5py.File(file2use, 'r')
        # print(list(F.keys())) #'DATA', 'RESPONSE', 'RESPONSETIME', 'trialtype'

        ### get variables of interest
        # Feedback
        feedback[:,:,day_count] = np.array(F['PREDICTIONS_Structured_ALL'])
        predictedstate[:,:,day_count]  = np.array(F['PREDICTIONS_Structured'])

        # Experiment Settings

        moveonsets[:,:,day_count]  = np.array(F['DATA']['MOVEONSETS'])


    # Plot histograms of feeback recieved per day

    feedback_proportion = np.empty((4, settings.num_days))
    fig, (ax) = plt.subplots(1, 3, figsize=(16, 6))
    bins = [0.5,1.5,2.5,3.5,4.5,5.5]
    bins2 = [0.5, 1.5, 2.5, 3.5, 4.5]
    for day_count, day_val in enumerate(settings.daysuse):
        counts, bins = np.histogram(feedback[:,:,day_count], bins=bins)
        counts = (counts / np.sum(counts)) * settings.trialduration
        counts = np.delete(counts, 2)
        feedback_proportion[:,day_count] = counts

        # plot
        ax[day_count].hist(bins2[:-1], bins2, weights=counts, edgecolor = 'black', facecolor=settings.lightteal)
        ax[day_count].set_title('day ' + str(day_val))
        ax[day_count].set_ylim([0, 4])
        ax[day_count].set_ylabel('count')
        ax[day_count].set_xlabel('Feedback')
        ax[day_count].spines['top'].set_visible(False)
        ax[day_count].spines['right'].set_visible(False)

    titlestring = bids.substring + ' neurofeedback histogram by day ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # create simplified feedback
    feedback_simple = feedback.copy()
    feedback_simple[feedback == 2] = 1
    feedback_simple[feedback == 3] = 0
    feedback_simple[feedback == 4] = 2
    feedback_simple[feedback == 5] = 2

    def runs_of_values_array(dat, value):
        bits = np.zeros(dat.shape)
        bits[np.where(dat == value)] = 1

        # make sure all runs of ones are well-bounded
        bounded = np.hstack(([0], bits, [0]))
        # get 1 at run starts and -1 at run ends
        difs = np.diff(bounded)
        run_starts, = np.where(difs > 0)
        run_ends, = np.where(difs < 0)
        return run_ends - run_starts

    # duration in state
    runs_correct_all = {}
    runs_incorrect_all = {}
    numruns = np.empty((2, settings.num_days, settings.num_trials))
    runlength = np.empty((2, settings.num_days))

    for day_count, day_val in enumerate(settings.daysuse):
        runs_correct = []
        runs_incorrect = []

        for TT in np.arange(settings.num_trials):
            print(day_val, TT)
            dat = np.squeeze(feedback_simple[TT,:,day_count])

            correct =  runs_of_values_array(dat, value=1)
            incorrect = runs_of_values_array(dat, value=2)
            runs_correct = np.concatenate((runs_correct, correct))
            runs_incorrect = np.concatenate((runs_incorrect,incorrect))

            numruns[0, day_count, TT] = len(correct)
            numruns[1, day_count, TT] = len(incorrect)

        # store for later use
        runs_correct_all[settings.string_testday[day_count]] = runs_correct
        runs_incorrect_all[settings.string_testday[day_count]] = runs_incorrect

        runlength[0, day_count] = np.mean(runs_correct)
        runlength[1, day_count] = np.mean(runs_incorrect)

    numruns_short = np.mean(numruns,2)

    ## Summarise with a plot
    fig, (ax) = plt.subplots(1, 2, figsize=(12, 6))

    labels = ['Day 1', 'Day 2', 'Day 3']
    x = np.arange(len(labels))  # the label locations
    width = 0.35  # the width of the bars

    # plot
    rects1 = ax[0].bar(x - width / 2, numruns_short[0,:], width, label='Correct')
    rects2 = ax[0].bar(x + width / 2, numruns_short[1,:], width, label='Incorrect')
    ax[0].set_xticks(x)
    ax[0].set_xticklabels(labels)
    ax[0].set_title('# of runs/ trial')
    ax[0].set_ylabel('# of runs')
    ax[0].set_xlabel('Test day')
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)
    ax[0].legend()

    rects1 = ax[1].bar(x - width / 2, runlength[0, :]/settings.mon_ref, width, label='Correct')
    rects2 = ax[1].bar(x + width / 2, runlength[1, :]/settings.mon_ref, width, label='Incorrect')
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels)
    ax[1].set_title('Run Length')
    ax[1].set_ylabel('Run Lengt (s)')
    ax[1].set_xlabel('Test day')
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    ax[1].legend()

    titlestring = bids.substring + ' Runs of feedback type ' + settings.string_attntrained[
        settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # Get Feedback ERP
    # create timelims
    timelims = [-1 * settings.mon_ref, 3 * settings.mon_ref]
    n_datapoints = len(np.arange(timelims[0], timelims[1]))

    feedbackERP = np.empty((n_datapoints, settings.num_days, settings.num_trials*settings.num_movements))
    feedbackERP[:] = np.nan
    for day_count, day_val in enumerate(settings.daysuse):
        count = -1
        for TT in np.arange(settings.num_trials):
            count = count + 1
            MO = moveonsets[:, TT, day_count].astype(int)
            trialdat = np.squeeze(feedback[TT,:,day_count])

            # create empty starting template
            tmp = np.zeros(n_datapoints)
            tmp[:] = np.nan

            for movecount in np.arange(settings.num_movements):
                start = MO[movecount] + timelims[0]
                stop = MO[movecount] + timelims[1]

                # get data
                if stop >= len(trialdat):
                    feedbackERP[0 : len(trialdat[start : -1]), day_count, count] = trialdat[start : -1]
                else:
                    feedbackERP[:,day_count, count] = trialdat[start : stop]


    plt.figure()
    datplot = np.nanmean(feedbackERP, 2)
    plt.plot(datplot)
    plt.vlines(144, 2.4, 3.4)
    plt.vlines(216, 2.4, 3.4)

    np.savez(bids.direct_results / Path(bids.substring + "NeurofeedbackSummaries"),
             feedback_proportion=feedback_proportion,
             runlength=runlength,
             numruns=numruns_short,
             feedbackERP=feedbackERP)




