# Import nescescary packages
import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
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


    plt.figure()
    plt.vlines(144, 0, 6)
    plt.plot(feedback[50, :, 2], color='r') #50
    plt.ylim([0, 6])







    np.savez(bids.direct_results / Path(bids.substring + "NeurofeedbackSummaries"),
             feedback_proportion=feedback_proportion,
             runlength=runlength,
             numruns=numruns_short,
             feedbackERP=feedbackERP)


def collate_Neurofeedback(settings):
    print('collating Neurofeedback for Space Vs. Feat Training')
    settings = settings.get_settings_behave_duringNF()

    # pre-allocate
    timelims = [-1 * settings.mon_ref, 3 * settings.mon_ref]
    n_datapoints = len(np.arange(timelims[0], timelims[1]))

    feedbackERPALL = np.empty((n_datapoints, settings.num_days, settings.num_attnstates))
    feedbackERPALL_error = np.empty((n_datapoints, settings.num_days, settings.num_attnstates))

    string_correct = ["correct", "incorrect"]
    string_feedback = ["Sustained Correct", "Correct", "Incorrect", "Sustained Incorrect"]

    numruns = []
    runlengths = []
    daystrings = []
    correctstrings = []
    attnstrings = []
    substrings = []
    substrings_fb = []
    daystrings_fb = []
    feedbackstrings = []
    attnstrings_fb = []
    feedback = []

    # cycle trough space and feature train groups
    for attntrained in np.arange(settings.num_attnstates):  # cycle trough space and feature train groups

        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_behave_duringNF()

        # pre-allocate
        feedbackERPcond = np.empty(
            (n_datapoints, settings.num_days, settings.num_subs, settings.num_trials * settings.num_movements))
        numrunscond = np.empty((2, settings.num_days, settings.num_subs))
        runlengthcond = np.empty((2, settings.num_days, settings.num_subs))
        feedbackproportioncond = np.empty((4, settings.num_days, settings.num_subs))

        substring_short = []

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "NeurofeedbackSummaries.npz"),
                              allow_pickle=True)

            runlengthcond[:, :, sub_count] = results["runlength"] / settings.mon_ref
            numrunscond[:, :, sub_count] = results["numruns"]
            feedbackERPcond[:, :, sub_count, :] = results["feedbackERP"]
            feedbackproportioncond[:, :, sub_count] = results["feedback_proportion"]

            substring_short = np.concatenate((substring_short, [bids.substring]))

        # summarise feedback ERP
        feedbackERPALL[:, :, attntrained] = np.nanmean(np.nanmean(feedbackERPcond, 3), 2)

        tmp = np.nanmean(feedbackERPcond, 3)
        suberr = np.nanmean(tmp, 0)
        granderr = np.nanmean(suberr, 1)
        x = tmp - (suberr - np.tile(granderr, (settings.num_subs, 1)).T)
        feedbackERPALL_error[:, :, attntrained] = np.nanstd(x) / np.sqrt(settings.num_subs)

        # Summarise runs
        for attentiontype in np.arange(2):
            for testday in np.arange(settings.num_days):
                substrings = np.concatenate((substrings, substring_short))
                daystrings = np.concatenate((daystrings, [settings.string_testday[testday]] * settings.num_subs))
                correctstrings = np.concatenate((correctstrings, [string_correct[attentiontype]] * settings.num_subs))
                attnstrings = np.concatenate(
                    (attnstrings, [settings.string_attntrained[attntrained]] * settings.num_subs))

                runlengths = np.concatenate((runlengths, runlengthcond[attentiontype, testday, :]))
                numruns = np.concatenate((numruns, numrunscond[attentiontype, testday, :]))

        print(np.mean(feedbackproportioncond, 2))

        # Summarise feedbackproportion
        for feedbacktype in np.arange(4):
            for testday in np.arange(settings.num_days):
                substrings_fb = np.concatenate((substrings_fb, substring_short))
                daystrings_fb = np.concatenate((daystrings_fb, [settings.string_testday[testday]] * settings.num_subs))
                feedbackstrings = np.concatenate((feedbackstrings, [string_feedback[feedbacktype]] * settings.num_subs))
                attnstrings_fb = np.concatenate(
                    (attnstrings_fb, [settings.string_attntrained[attntrained]] * settings.num_subs))

                feedback = np.concatenate((feedback, feedbackproportioncond[feedbacktype, testday, :]))

    data = {'SubID': substrings, 'Testday': daystrings, 'Attentionstate': correctstrings,
            'AttentionTrained': attnstrings,
            'RunLengths': runlengths, 'NumberofRuns': numruns}
    df_runs = pd.DataFrame(data)

    df_runs.loc[(np.abs(stats.zscore(df_runs['RunLengths'])) > 3), ["RunLengths", "NumberofRuns"]] = np.nan

    data = {'SubID': substrings_fb, 'Testday': daystrings_fb, 'FeedbackType': feedbackstrings,
            'AttentionTrained': attnstrings_fb,
            'FeedbackProportion (s)': feedback}
    df_feedback = pd.DataFrame(data)

    ## Plot results - run lengths
    fig, (ax1) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    for attn, attnstring in enumerate(settings.string_attntrained):
        sns.violinplot(x="Testday", y="RunLengths", hue="Attentionstate",
                       data=df_runs[df_runs["AttentionTrained"].isin([attnstring])],
                       palette=sns.color_palette(colors), style="ticks", ax=ax1[attn], split=True, inner="stick")

        ax1[attn].spines['top'].set_visible(False)
        ax1[attn].spines['right'].set_visible(False)

        ax1[attn].set_title(attnstring)
        ax1[attn].set_ylim([0.2, 0.9])
        ax1[attn].set_ylabel("Time attended / trial (s)")

    titlestring = 'Attention States categorised Time per trial'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    ## Plot results - number of runs
    fig, (ax1) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    for attn, attnstring in enumerate(settings.string_attntrained):
        sns.violinplot(x="Testday", y="NumberofRuns", hue="Attentionstate",
                       data=df_runs[df_runs["AttentionTrained"].isin([attnstring])],
                       palette=sns.color_palette(colors), style="ticks", ax=ax1[attn], split=True, inner="stick")

        ax1[attn].spines['top'].set_visible(False)
        ax1[attn].spines['right'].set_visible(False)

        ax1[attn].set_title(attnstring)
        ax1[attn].set_ylim([4, 14])
        ax1[attn].set_ylabel("Time attended / trial (s)")

    titlestring = 'Attention States categorised Number of runs per trial'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    ## Plot results - feedback
    fig, (ax1) = plt.subplots(1, 3, figsize=(18, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    for day, daystring in enumerate(settings.string_testday[0:3]):
        sns.violinplot(x="FeedbackType", y="FeedbackProportion (s)", hue="AttentionTrained",
                       data=df_feedback[df_feedback["Testday"].isin([daystring])],
                       palette=sns.color_palette(colors), style="ticks", ax=ax1[day], split=True, inner="stick")

        ax1[day].spines['top'].set_visible(False)
        ax1[day].spines['right'].set_visible(False)

        ax1[day].set_title(daystring)
        ax1[day].set_ylim([0, 4])
        ax1[day].set_ylabel("Time attended / trial (s)")

    titlestring = 'Feedback proportions by training day'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # plot "ERPs" to motion epochs
    # feedbackERPALL[:, :, attntrained] = np.nanmean(np.nanmean(feedbackERPcond, 3), 2)
    # feedbackERPALL_error[:, :, attntrained] = np.nanstd(np.nanmean(feedbackERPcond, 3), 2)
    fig, (ax1) = plt.subplots(1, 2, figsize=(12, 6))
    colours = [settings.yellow, settings.lightteal, settings.medteal]
    t = np.arange(timelims[0], timelims[1]) / settings.mon_ref

    for attn, attnstring in enumerate(settings.string_attntrained):
        for day, daystring in enumerate(settings.string_testday[0:3]):
            datplot = feedbackERPALL[:, day, attn] - np.nanmean(feedbackERPALL[0:settings.mon_ref, day, attn], 0)
            ax1[attn].plot(t, datplot, color=colours[day])
            ax1[attn].fill_between(t, datplot - feedbackERPALL_error[:, day, attn],
                                   datplot + feedbackERPALL_error[:, day, attn], facecolor=colours[day], alpha=0.2)

        ax1[attn].vlines([0, 0.5], -1, 1, 'k')

        ax1[attn].set_xlim([-1, 2])
        ax1[attn].set_title(attnstring)
        ax1[attn].set_xlabel('Time relative to movement (s)')
        ax1[attn].legend(settings.string_testday[0:3])

    ax1[0].set_ylim([-0.1, 0.1])
    ax1[1].set_ylim([-0.1, 0.1])

    titlestring = 'Neurofeedback motion ERPs'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    motdiff = - np.mean(feedbackERPALL[np.logical_and(t > -0.5, t < 0), :, :], 0) + np.mean(
        feedbackERPALL[np.logical_and(t > 1.5, t < 2.0), :, :], 0)




