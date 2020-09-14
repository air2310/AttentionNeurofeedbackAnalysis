# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import h5py
import matplotlib.pyplot as plt
import Analysis_Code.helperfunctions_ATTNNF as helper
import Analysis_Code.functions_getEEGprepost as geegpp
import Analysis_Code.functions_getEEG_duringNF as geegdnf
import Analysis_Code.analyse_visualsearchtask as avissearch
import Analysis_Code.analyse_nbacktask as anback
# TODO:

# correlate behaviour with ssvep selectivity
# look at differences between classifiable and unclassifiable participants.
# check prepost differences in nback task
# analyse feedback - how long in each state? how does it correspond to behaviour?
# analyse behaviour during neurofeedback training - across the three days.
# look at withing session learning curves for SSVEPs

# asthetic changes to plotting - add subject scatterpoints to behavioural results
# stats on behaviour
# integrate behavioural analyses with python
# group average topoplots

# collate nbacktask

# Decide which analyses to do
# analyseEEGprepost =True # analyse EEG Pre Vs. Post Training
# analyseEEG_duringNF = True # analyse EEG during Neurofeedback
# analyse_visualsearchtask = True # Analyse Visual Search Task
# analyse_nbacktask = True # Analyse N-back Task
#
collateEEGprepost = False# Collate EEG Pre Vs. Post Training across subjects
collateEEG_duringNF = False # Collate EEG during Neurofeedback
collate_visualsearchtask = False # Collate Visual Search results
collate_nbacktask = True # Analyse N-back Task

analyseEEGprepost =False # analyse EEG Pre Vs. Post Training
analyseEEG_duringNF = False # analyse EEG during Neurofeedback
analyse_visualsearchtask = False # Analyse Visual Search Task
analyse_nbacktask = False # Analyse N-back Task

# collateEEGprepost = True # Collate EEG Pre Vs. Post Training across subjects
# collateEEG_duringNF = True # Collate EEG during Neurofeedback
# collate_visualsearchtask = True # Collate Visual Search results

# setup generic settings
attntrained = 1 # ["Space", "Feature"]
settings = helper.SetupMetaData(attntrained)

print("Analysing Data for condition train: " + settings.string_attntrained[settings.attntrained])

# iterate through subjects for individual subject analyses
for sub_count, sub_val in enumerate(settings.subsIDX):
    if (analyseEEGprepost):
        geegpp.analyseEEGprepost(settings, sub_val)

    if (analyseEEG_duringNF):
        geegdnf.analyseEEG_duringNF(settings, sub_val)

    if (analyse_visualsearchtask):
        avissearch.analyse_visualsearchtask(settings, sub_val)

    if (analyse_nbacktask):
        if (sub_val != 21):
            anback.analyse_nbacktask(settings, sub_val)

# Collate EEG prepost
if (collateEEGprepost):

    print('collating SSVEP amplitudes pre Vs. post training')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    fftdat_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    wavelets_prepost_group = np.empty((len(timepoints_zp) + 1, settings.num_days, settings.num_attd_unattd, settings.num_attnstates, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_pre_post_results.npz"), allow_pickle=True) #
        #saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

        # store results
        SSVEPs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean']
        SSVEPs_epochs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']
        fftdat_group[:,:,:,:,:,sub_count] = results['fftdat']
        fftdat_epochs_group[:, :, :, :, :, sub_count] = results['fftdat_epochs']
        wavelets_prepost_group[:,:,:,:,sub_count] = results['wavelets_prepost']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

    # plot grand average frequency spectrum
    fftdat_ave = np.mean(fftdat_group, axis = 5)
    geegpp.plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring='ERP', settings=settings, freq=freq)

    fftdat_epochs_ave = np.mean(fftdat_epochs_group, axis = 5)
    geegpp.plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    geegpp.plotGroupSSVEPsprepost(SSVEPs_prepost_group, bids, ERPstring='ERP', settings=settings)
    geegpp.plotGroupSSVEPsprepost(SSVEPs_epochs_prepost_group, bids, ERPstring='Single Trial', settings=settings)

    # plot wavelet results
    wavelets_prepost_ave = np.mean(wavelets_prepost_group, axis=4)
    wavelets_prepost_std = np.std(wavelets_prepost_group, axis=4)/num_subs

    # plot wavelet data
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 15))
    for attn in np.arange(settings.num_attnstates):
        for dayuse in np.arange(settings.num_days):
            if (dayuse == 0): axuse = ax1[attn]
            if (dayuse == 1): axuse = ax2[attn]
            if (attn == 0):
                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 0, attn] - wavelets_prepost_std[:, dayuse, 0, attn],
                                   wavelets_prepost_ave[:, dayuse, 0, attn] + wavelets_prepost_std[:, dayuse, 0, attn], alpha=0.3, facecolor=settings.medteal)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 0, attn], color=settings.medteal,
                           label=settings.string_attd_unattd[0])

                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 1, attn] - wavelets_prepost_std[:, dayuse, 1, attn],
                                   wavelets_prepost_ave[:, dayuse, 1, attn] + wavelets_prepost_std[:, dayuse, 1, attn], alpha=0.3, facecolor=settings.medteal)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 1, attn], color=settings.lightteal,
                           label=settings.string_attd_unattd[1])
            else:
                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 0, attn] - wavelets_prepost_std[:, dayuse, 0, attn],
                                   wavelets_prepost_ave[:, dayuse, 0, attn] + wavelets_prepost_std[:, dayuse, 0, attn],
                                   alpha=0.3, facecolor=settings.orange)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 0, attn], color=settings.orange,
                           label=settings.string_attd_unattd[0])

                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 1, attn] - wavelets_prepost_std[:, dayuse, 1, attn],
                                   wavelets_prepost_ave[:, dayuse, 1, attn] + wavelets_prepost_std[:, dayuse, 1, attn],
                                   alpha=0.3, facecolor=settings.yellow)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 1, attn], color=settings.yellow,
                           label=settings.string_attd_unattd[1])
            axuse.set_xlim(-1, 6)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.set_ylim(100, 1000)
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn] + ' ' + settings.string_prepost[dayuse])

    titlestring = 'Group Mean wavelets pre-post ' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # save attentional selectivity for stats - single trial
    ssvep_selectivity_prepost = SSVEPs_epochs_prepost_group[0, :, :, :] - SSVEPs_epochs_prepost_group[1, :, :, :]
    tmp = np.reshape(ssvep_selectivity_prepost, (4,settings.num_subs)) # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost_epochs.npy"), tmp)

    # save attentional selectivity for stats
    ssvep_selectivity_prepost = SSVEPs_prepost_group[0, :, :, :] - SSVEPs_prepost_group[1, :, :, :]
    tmp = np.reshape(ssvep_selectivity_prepost,
                     (4, settings.num_subs))  # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost.npy"), tmp)

# Collate EEG prepost
if (collateEEG_duringNF):
    print('collating SSVEP amplitudes during NF')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_duringNF()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    fftdat_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_duringNF_results.npz"), allow_pickle=True) #
        #saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

        # store results
        SSVEPs_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean']
        SSVEPs_epochs_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']
        fftdat_group[:,:,:,:,:,sub_count] = results['fftdat']
        fftdat_epochs_group[:, :, :, :, :, sub_count] = results['fftdat_epochs']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

    # plot grand average frequency spectrum
    fftdat_ave = np.nanmean(fftdat_group, axis = 5)
    geegdnf.plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring='ERP', settings=settings, freq=freq)

    fftdat_epochs_ave = np.nanmean(fftdat_epochs_group, axis = 5)
    geegdnf.plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    geegdnf.plotGroupSSVEPs(SSVEPs_group, bids, ERPstring='ERP', settings=settings)
    geegdnf.plotGroupSSVEPs(SSVEPs_epochs_group, bids, ERPstring='Single Trial', settings=settings)

# Collate Visual Search Task
if (collate_visualsearchtask):
    print('Collating Visual Search Task')

    # get task specific settings
    settings = settings.get_settings_visualsearchtask()

    # preallocate group mean variables
    num_subs = settings.num_subs
    acc_vissearch_all = np.empty((settings.num_trialscond, settings.num_setsizes, settings.num_days, num_subs))
    rt_vissearch_all = np.empty((settings.num_trialscond, settings.num_setsizes, settings.num_days, num_subs))
    mean_acc_all = np.empty((settings.num_setsizes, settings.num_days, num_subs))
    mean_rt_all = np.empty((settings.num_setsizes, settings.num_days, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "visual_search_results.npz"), allow_pickle=True)  #
        # saved vars: meanacc=meanacc, meanrt=meanrt, acc_vissearch=acc_vissearch, rt_vissearch=rt_vissearch

        # store results
        acc_vissearch_all[ :, :, :, sub_count] = results['acc_vissearch']
        mean_acc_all[ :, :, sub_count] = results['meanacc']

        rt_vissearch_all[ :, :, :, sub_count] = results['rt_vissearch']
        mean_rt_all[ :, :, sub_count] = results['meanrt']


    # plot accuracy results
    meanacc = np.nanmean(mean_acc_all, axis=2)
    erroracc = np.empty((settings.num_setsizes, settings.num_days))
    erroracc[:, 0] = helper.within_subjects_error(np.squeeze(mean_acc_all[:,0,:]).T)
    erroracc[:, 1] = helper.within_subjects_error(np.squeeze(mean_acc_all[:,1,:]).T)

    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_prepost
    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, meanacc[0, :], width, yerr=erroracc[0, :], label=settings.string_setsize[0], facecolor=settings.lightteal)
    plt.bar(x, meanacc[1, :], width,  yerr=erroracc[1, :],label=settings.string_setsize[1], facecolor=settings.medteal)
    plt.bar(x + width, meanacc[2, :], width,yerr=erroracc[2, :],  label=settings.string_setsize[2], facecolor=settings.darkteal)

    plt.ylim([90, 100])
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = 'Visual Search Accuracy train ' + settings.string_attntrained[settings.attntrained]
    plt.title(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    # plot reaction time results
    meanrt = np.nanmean(mean_rt_all, axis=2)
    errorrt = np.empty((settings.num_setsizes, settings.num_days))
    errorrt[:, 0] = helper.within_subjects_error(np.squeeze(mean_rt_all[:,0,:]).T)
    errorrt[:, 1] = helper.within_subjects_error(np.squeeze(mean_rt_all[:,1,:]).T)

    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_prepost
    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, meanrt[0, :], width, yerr=errorrt[0, :], label=settings.string_setsize[0], facecolor=settings.lightteal)
    plt.bar(x, meanrt[1, :], width,  yerr=errorrt[1, :],label=settings.string_setsize[1], facecolor=settings.medteal)
    plt.bar(x + width, meanrt[2, :], width,yerr=errorrt[2, :],  label=settings.string_setsize[2], facecolor=settings.darkteal)

    plt.ylabel('reaction time (s)')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = 'Visual Search reaction time train ' + settings.string_attntrained[settings.attntrained]
    plt.title(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

if (collate_nbacktask):
    import seaborn as sns
    import pandas as pd

    # preallocate
    num_subs = np.zeros((settings.num_attnstates))
    daystrings = []
    attnstrings = []
    accuracy_compare = []
    rt_compare = []

    print('Collating N-back Task for space and feature train')

    for attntrained in np.arange(settings.num_attnstates): # cycle trough space and feature train groups

        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_nbacktask()

        # pre-allocate for this group
        if (attntrained == 1): # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate, settings.subsIDXcollate==21)
            settings.num_subs = settings.num_subs-1

        num_subs[attntrained] = settings.num_subs
        mean_acc_all = np.empty((settings.num_days, settings.num_subs))
        mean_rt_all = np.empty((settings.num_days, settings.num_subs))

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):

            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "Nback_results.npz"),
                              allow_pickle=True)  #saved vars: meanacc=meanacc, meanrt=meanrt, acc_vissearch=acc_nback, rt_vissearch=rt_nback

            # store results temporarily
            mean_acc_all[:, sub_count] = results['meanacc']*100
            mean_rt_all[:, sub_count] = results['meanrt']

        # store results for attention condition
        tmp = [settings.string_prepost[0]] * settings.num_subs + [settings.string_prepost[1]] * settings.num_subs
        daystrings = np.concatenate((daystrings, tmp))

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days
        attnstrings = np.concatenate((attnstrings, tmp))

        tmp = np.concatenate((mean_acc_all[0, :], mean_acc_all[1, :]))
        accuracy_compare = np.concatenate((accuracy_compare, tmp))

        tmp = np.concatenate((mean_rt_all[0, :], mean_rt_all[1, :]))
        rt_compare = np.concatenate((rt_compare, tmp))

    # create the data frames for accuracy and reaction time data
    data = {'Testday': daystrings, 'Attention Trained': attnstrings, 'Accuracy (%)': accuracy_compare }
    df_acc = pd.DataFrame(data)

    data = {'Testday': daystrings, 'Attention Trained': attnstrings, 'Reaction Time (s)': rt_compare}
    df_rt = pd.DataFrame(data)

    # plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = ["#112F41", "#4CB99F"]

    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y= "Accuracy (%)" , hue = "Testday", data=df_acc, palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title("Accuracy")

    # Reaction time Grouped violinplot

    sns.violinplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", data=df_rt,
                   palette=sns.color_palette(colors), style="ticks", ax=ax2, split=True)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title("Reaction time")

    titlestring = 'Nback Results Compare Training'

    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

