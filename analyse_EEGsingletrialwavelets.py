import numpy as np
from pathlib import Path
import mne
import helperfunctions_ATTNNF as helper
import matplotlib as mpl
mpl.use('TkAgg')  # or can use 'TkAgg', whatever you have/prefer
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools


def analyseEEGprepost(settings, sub_val):
    print('analysing SSVEP amplitudes pre Vs. post training')

    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate
    num_epochs = settings.num_trials / settings.num_conditions
    epochs_days = np.empty((int(num_epochs), settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels,settings.num_days))
    epochs_days[:] = np.nan

    # iterate through test days to get data
    for day_count, day_val in enumerate(settings.daysuse):

        print(day_val)
        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # get EEG data
        raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count, day_val, settings)

        # Epoch to events of interest
        event_id = {'Feat/Black': 121, 'Feat/White': 122,
                    'Space/Left_diag': 123, 'Space/Right_diag': 124}  # will be different triggers for training days

        epochs = mne.Epochs(eeg_data_interp, events, event_id=event_id, tmin=settings.timelimits_zeropad[0],
                            tmax=settings.timelimits_zeropad[1],
                            baseline=(0, 1 / settings.samplingfreq), picks=np.arange(settings.num_electrodes),
                            reject=dict(eeg=400), detrend=1)  #

        # epoching plan for wavelets
        # idx = events[:, 2] ismember eventid
        # lim_min = idx + timelimits 0
        # lim_max = idx + timelimits 1
        # for ii = 1: len(idx)
        #   shorterevents = events[limmin:limax, 0]
        #   find shorterevents ismember motioneventid.

        # drop bad channels
        epochs.drop_bad()
        # epochs.plot_drop_log()
        # epochs2 = epochs.equalize_event_counts(event_id, method='mintime')

        # visualise topo
        # epochs.plot_psd_topomap()

        # get data for each  condition
        epochs_days[0:sum(elem == [] for elem in epochs['Space/Left_diag'].drop_log), :, :, 0, 0, day_count] = epochs[
            'Space/Left_diag'].get_data()
        epochs_days[0:sum(elem == [] for elem in epochs['Space/Right_diag'].drop_log), :, :, 0, 1, day_count] = epochs[
            'Space/Right_diag'].get_data()
        epochs_days[0:sum(elem == [] for elem in epochs['Feat/Black'].drop_log), :, :, 1, 0, day_count] = epochs[
            'Feat/Black'].get_data()
        epochs_days[0:sum(elem == [] for elem in epochs['Feat/White'].drop_log), :, :, 1, 1, day_count] = epochs[
            'Feat/White'].get_data()

    # Get Best electrodes
    erps_days = np.squeeze(np.nanmean(epochs_days, axis=0))
    fftdat, fftdat_epochs, freq = getSSVEPs(erps_days, epochs_days, epochs, settings, bids)
    fftdat_epochs = np.nanmean(fftdat_epochs, axis=0)  # average across trials to get the same shape for single trial SSVEPs
    SSVEPs_prepost_epochs, SSVEPs_prepost_channelmean_epochs, BEST = getSSVEPS_conditions(settings, fftdat_epochs, freq)  # single trial

    # get wavelets
    runlength, numruns = get_wavelets_prepost(epochs_days, settings, epochs, BEST)

    np.savez(bids.direct_results / Path("EEGResults_sustainedattention"),
                 runlength=runlength,
                 numruns=numruns)


def get_wavelets_prepost(epochs_days, settings, epochs, BEST):

    erps_days_wave = epochs_days.transpose(0, 3, 4, 5, 1, 2)  # [cuetype, level, day, chans,time,]
    num_epochs = erps_days_wave.shape[0]
    # Get wavelets for each frequency
    wavelets = np.empty((num_epochs, settings.num_attnstates, settings.num_levels, len(epochs.times), settings.num_spaces, settings.num_features, settings.num_days))
    for level in np.arange(2):
        for space_count, space in enumerate(['Left_diag', 'Right_diag']):
            for feat_count, feat in enumerate(['Black', 'White']):
                for day_count, day_val in enumerate(settings.daysuse):
                    bestuse = BEST[:, space_count, feat_count, day_count].astype(int)
                    hzuse = settings.hz_attn[space_count, feat_count]
                    datuse = np.mean(erps_days_wave[:, :, level, day_count, bestuse, :], axis=2)
                    wavelets[:, :, level, :, space_count, feat_count, day_count] = np.squeeze(mne.time_frequency.tfr_array_morlet(datuse, settings.samplingfreq, freqs=[hzuse],n_cycles=[hzuse], output='power')) / len(epochs.times)


    # crop
    zerotimes = np.where(np.logical_and(epochs.times > settings.timelimits[0], epochs.times < settings.timelimits[1]))
    wavelets = wavelets[:, :, :, zerotimes[0], :, :, :]
    times2 = epochs.times[zerotimes[0]]

    # Space wavelets
    cuetype = 0  # space
    left_diag, right_diag = 0, 1

    # spaceSSVEPs = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_levels))
    spacewavelets = np.empty((num_epochs, len(times2), settings.num_days, settings.num_attd_unattd, settings.num_levels))
    for level_count, level in enumerate(['Left_diag', 'Right_diag']):  # cycle through space trials on which the left and right diag were cued
        if (level == 'Left_diag'):  # when left diag cued
            attendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, left_diag, :, :], 2)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, right_diag, :, :], 2) # average across right_diag frequencies at both features (black, white)

        if (level == 'Right_diag'):  # when right diag cued
            attendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, right_diag, :, :], 2)  # average across right_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, left_diag, :, :], 2)  # average across left_diag frequencies at both features (black, white)

        spacewavelets[:, :, :, 0, level_count] = attendedSSVEPs
        spacewavelets[:, :, :, 1, level_count] = unattendedSSVEPs


    # feature wavelets
    cuetype = 1  # space
    black, white = 0, 1

    featurewavelets = np.empty((num_epochs, len(times2), settings.num_days, settings.num_attd_unattd, settings.num_levels))
    for level_count, level in enumerate(['Black', 'White']):  # cycle through space trials on which the black and white were cued
        if (level == 'Black'):  # when left diag cued
            attendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, :, black, :], 2)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, :, white, :], 2)  # average across right_diag frequencies at both features (black, white)

        if (level == 'White'):  # when right diag cued
            attendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, :, white, :], 2)  # average across right_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[:, cuetype, level_count, :, :, black, :], 2)  # average across left_diag frequencies at both features (black, white)

        featurewavelets[:, :, :, 0, level_count] = attendedSSVEPs # Epochs, time, days, attdunattd, levels
        featurewavelets[:, :, :, 1, level_count] = unattendedSSVEPs

    # Get selectivity
    spacewave_select = spacewavelets[:, :, :, 0, :] - spacewavelets[:, :, :, 1, :] # Epochs, time, days, levels
    featwave_select = featurewavelets[:, :, :, 0, :] - featurewavelets[:, :, :, 1, :]  # Epochs, time, days, levels


    # Define run counting function
    def runs_of_ones_list(bits):
        return [sum(g) for b, g in itertools.groupby(bits) if b]

    # Get proportion of time in each state
    spacebits = spacewave_select > 0
    featbits = featwave_select > 0

    numruns = np.empty((settings.num_attnstates, num_epochs, settings.num_levels, settings.num_days))
    runlength = np.empty((settings.num_attd_unattd, settings.num_attnstates, settings.num_days))
    numruns[:] = np.nan
    runlength[:] = np.nan

    for day_count, day_val in enumerate(['pre-training', 'post-training']):
        space_cued = list()
        space_uncued = list()
        feat_cued = list()
        feat_uncued = list()

        for level in np.arange(2):
            for epoch in np.arange(num_epochs):
                space_cued.extend(runs_of_ones_list(spacebits[epoch, :, day_count, level]))
                space_uncued.extend(runs_of_ones_list(~spacebits[epoch, :, day_count, level]))
                feat_cued.extend(runs_of_ones_list(featbits[epoch, :, day_count, level]))
                feat_uncued.extend(runs_of_ones_list(~featbits[epoch, :, day_count, level]))

                numruns[0, epoch, level, day_count] = len(runs_of_ones_list(spacebits[epoch, :, day_count, level])) + len(runs_of_ones_list(~spacebits[epoch, :, day_count, level]))
                numruns[1, epoch, level, day_count] = len(runs_of_ones_list(featbits[epoch, :, day_count, level])) + len(runs_of_ones_list(~featbits[epoch, :, day_count, level]))

        runlength[0, 0, day_count] = np.mean(space_cued)/ settings.samplingfreq
        runlength[1, 0, day_count] = np.mean(space_uncued)/ settings.samplingfreq
        runlength[0, 1, day_count] = np.mean(feat_cued)/ settings.samplingfreq
        runlength[1, 1, day_count] = np.mean(feat_uncued)/ settings.samplingfreq

    # Plot results
    fig, ax = plt.subplots(2, 2, figsize=(10, 6))
    for cue, cuestring in enumerate(settings.string_cuetype):
        for day_count, day_val in enumerate(['pre-training', 'post-training']):
            ax[day_count, cue].bar([1, 2], runlength[:, cue, day_count])
            ax[day_count, cue].set_ylabel('Run Length (s)')
            ax[day_count, cue].set_xticks([1,2])
            ax[day_count, cue].set_xticklabels(["Cued", "Uncued"])
            ax[day_count, cue].set_frame_on(False)
            ax[day_count, cue].set_title(cuestring + " " + day_val)


    return runlength, numruns


def getSSVEPs(erps_days, epochs_days, epochs, settings, bids):
    from scipy.fft import fft, fftfreq, fftshift
    import matplotlib.pyplot as plt

    # for fft - zeropad - old wat
    # zerotimes = np.where(
    #     np.logical_or(epochs.times < settings.timelimits[0], epochs.times > settings.timelimits[1]))
    # erps_days[:, zerotimes, :, :, :] = 0
    # epochs_days[:, :, zerotimes, :, :, :] = 0
    #
    # # for fft - fft
    # fftdat = np.abs(fft(erps_days, axis=1)) / len(epochs.times)
    # fftdat_epochs = np.abs(fft(epochs_days, axis=2)) / len(epochs.times)
    #

    # crop
    zerotimes = np.where(np.logical_and(epochs.times > settings.timelimits[0], epochs.times < settings.timelimits[1]))
    erps_days2 = erps_days[:, zerotimes[0], :, :, :]
    epochs_days2 = epochs_days[:, :, zerotimes[0], :, :, :]
    times2 = epochs.times[zerotimes[0]]

    # for fft - fft
    fftdat = np.abs(fft(erps_days2, axis=1)) / len(times2)
    fftdat_epochs = np.abs(fft(epochs_days2, axis=2)) / len(times2)

    ## plot ERP FFT spectrum

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    freq = fftfreq(len(times2), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.mean(fftdat, axis=0)

    for day_count in np.arange(2):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Space/Left_diag', color=settings.lightteal)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Space/Right_diag', color=settings.medteal)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='Feat/Black', color=settings.darkteal)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='Feat/White', color=settings.yellow)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        # axuse.set_ylim(0, .5)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()

    titlestring = bids.substring + 'ERP FFT Spectrum'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # plot single trial FFT spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    freq = fftfreq(len(times2), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.nanmean(np.mean(fftdat_epochs, axis=1), axis=0)

    for day_count in np.arange(2):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Space/Left_diag', color=settings.lightteal)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Space/Right_diag', color=settings.medteal)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='Feat/Black', color=settings.darkteal)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='Feat/White', color=settings.yellow)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, 1)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()

    titlestring = bids.substring + 'Single Trial FFT Spectrum'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    return fftdat, fftdat_epochs, freq


def getSSVEPS_conditions(settings, fftdat, freq):
    # get indices for frequencies of interest
    hz_attn_index = np.empty((settings.num_spaces, settings.num_features))
    for space_count, space in enumerate(['Left_diag', 'Right_diag']):
        for feat_count, feat in enumerate(['Black', 'White']):
            hz_attn_index[space_count, feat_count] = np.argmin(np.abs(freq - settings.hz_attn[space_count, feat_count])) # ['\B \W'], ['/B /W']

    # Get best electrodes for each frequency
    BEST = np.empty((settings.num_best, settings.num_spaces, settings.num_features, settings.num_days))
    for space_count, space in enumerate(['Left_diag', 'Right_diag']):
        for feat_count, feat in enumerate(['Black', 'White']):
            for day_count, day_val in enumerate(settings.daysuse):
                tmp = np.mean(np.mean(fftdat[:, hz_attn_index[space_count, feat_count].astype(int), :, :, day_count], axis = 1), axis = 1) # average across cue conditions to get best electrodes for frequencies
                BEST[:,space_count, feat_count, day_count] = tmp.argsort()[-settings.num_best:]
                # print(BEST[:,space_count, feat_count, day_count])

    # Get SSVEPs for each frequency
    SSVEPs = np.empty((settings.num_attnstates, settings.num_levels, settings.num_spaces, settings.num_features, settings.num_days))
    SSVEPs_topo = np.empty((settings.num_electrodes,  settings.num_attnstates, settings.num_levels, settings.num_spaces, settings.num_features, settings.num_days))
    for space_count, space in enumerate(['Left_diag', 'Right_diag']):
        for feat_count, feat in enumerate(['Black', 'White']):
            for day_count, day_val in enumerate(settings.daysuse):
                for cuetype in np.arange(2):
                    for level in np.arange(2):
                        bestuse = BEST[:,space_count, feat_count, day_count].astype(int)
                        hzuse = hz_attn_index[space_count, feat_count].astype(int)

                        SSVEPs[cuetype, level, space_count, feat_count, day_count] = np.mean(fftdat[bestuse, hzuse, cuetype, level, day_count], axis=0)
                        SSVEPs_topo[:, cuetype, level, space_count, feat_count, day_count] = fftdat[:, hzuse, cuetype, level, day_count]

    # get ssveps for space condition, sorted to represent attended vs. unattended
    cuetype = 0  # space
    left_diag, right_diag =  0, 1

    spaceSSVEPs = np.empty(( settings.num_attd_unattd, settings.num_days, settings.num_levels))
    spaceSSVEPs_topo = np.empty((settings.num_electrodes, settings.num_attd_unattd, settings.num_days, settings.num_levels))
    for level_count, level in enumerate(['Left_diag', 'Right_diag']): # cycle through space trials on which the left and right diag were cued
        if level == 'Left_diag': # when left diag cued
            attendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, left_diag, :, :], 0) #  average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, right_diag, :, :], 0)  # average across right_diag frequencies at both features (black, white)

            attendedSSVEPs_topo = np.mean(SSVEPs_topo[:,cuetype, level_count, left_diag, :, :], 1)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs_topo = np.mean(SSVEPs_topo[:,cuetype, level_count, right_diag, :, :], 1)  # average across right_diag frequencies at both features (black, white)

        if level == 'Right_diag': # when right diag cued
            attendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, right_diag, :, :], 0)  # average across right_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, left_diag, :, :], 0)  # average across left_diag frequencies at both features (black, white)

            attendedSSVEPs_topo = np.mean(SSVEPs_topo[:, cuetype, level_count, right_diag, :, :], 1)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs_topo = np.mean(SSVEPs_topo[:, cuetype, level_count, left_diag, :, :], 1)  # average across right_diag frequencies at both features (black, white)

        spaceSSVEPs[0, :, level_count] = attendedSSVEPs
        spaceSSVEPs[1, :, level_count] = unattendedSSVEPs

        spaceSSVEPs_topo[:, 0, :,level_count] = attendedSSVEPs_topo
        spaceSSVEPs_topo[:, 1, :, level_count] = unattendedSSVEPs_topo

    # get ssveps for feature condition, sorted to represent attended vs. unattended
    cuetype = 1  # feature
    black, white = 0, 1
    featureSSVEPs = np.empty(( settings.num_attd_unattd, settings.num_days, settings.num_levels))
    featureSSVEPs_topo = np.empty((settings.num_electrodes,  settings.num_attd_unattd, settings.num_days,settings.num_levels))
    for level_count, level in enumerate(['Black', 'White']): # average through trials on which black and white were cued
        if (level == 'Black'):
            attendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, :, black, :], 0)  # average across black frequencies at both spatial positions
            unattendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, :, white, :], 0)  # average across white frequencies at both spatial positions

            attendedSSVEPs_topo = np.mean(SSVEPs_topo[:, cuetype, level_count, :, black, :], 1)  #  average across black frequencies at both spatial positions
            unattendedSSVEPs_topo = np.mean(SSVEPs_topo[:, cuetype, level_count, :, white, :],  1)  # average across white frequencies at both spatial positions

        if (level == 'White'):
            attendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, :, white, :], 0)  # average across white frequencies at both spatial positions
            unattendedSSVEPs = np.mean(SSVEPs[cuetype, level_count, :, black, :], 0)  # average across black frequencies at both spatial positions

            attendedSSVEPs_topo = np.mean(SSVEPs_topo[:, cuetype, level_count, :, white, :], 1)  # average across white frequencies at both spatial positions
            unattendedSSVEPs_topo = np.mean(SSVEPs_topo[:, cuetype, level_count, :, black, :], 1)  # average across blackfrequencies at both spatial positions

        featureSSVEPs[0, :, level_count] = attendedSSVEPs
        featureSSVEPs[1, :,level_count] = unattendedSSVEPs

        featureSSVEPs_topo[:, 0, :, level_count] = attendedSSVEPs_topo
        featureSSVEPs_topo[:, 1, :, level_count] = unattendedSSVEPs_topo

    # average across cue types and store the SSVEPs alltogether for plotting and further analysis
    SSVEPs_prepost_mean = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates))
    SSVEPs_prepost_mean[:, :, 0] = np.mean(spaceSSVEPs, axis=2)  # attd unattd, # daycount, # space/feat
    SSVEPs_prepost_mean[:, :, 1] = np.mean(featureSSVEPs, axis=2)

    # average across cue types and store the SSVEPs alltogether for plotting and further analysis (topos)
    SSVEPs_prepost = np.empty((settings.num_electrodes, settings.num_attd_unattd, settings.num_days,  settings.num_attnstates))
    SSVEPs_prepost[:, :, :, 0] = np.mean(spaceSSVEPs_topo, axis=3) # num electrodes # attd unattd, # daycount, # space/feat
    SSVEPs_prepost[:, :, :, 1] = np.mean(featureSSVEPs_topo, axis=3)

    return SSVEPs_prepost, SSVEPs_prepost_mean, BEST


def collate_singletrialEEG(settings):
    print('collating SSVEP amplitudes pre Vs. post training')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    traininggroup = list()
    SubID = list()
    SubIDval = list()
    testday = list()
    cuetype = list()
    attentionstate = list()
    runlength = list()


    for traininggroupcount, traininggroupstring in enumerate(settings.string_attntrained):
        # get task specific settings
        settings = helper.SetupMetaData(traininggroupcount)
        settings = settings.get_settings_EEG_prepost()

        # load results
        bids = helper.BIDS_FileNaming(0, settings, 1)

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path("EEGResults_sustainedattention.npz"), allow_pickle=True)  #
            # numruns = np.empty((settings.num_attnstates, num_epochs, settings.num_levels, settings.num_days))
            # numruns = results['numruns']

            # store results
            data = results['runlength']
            # runlength = np.empty((settings.num_attd_unattd, settings.num_attnstates, settings.num_days))

            for cuetypei , cuetypeval in enumerate(["Space", "Feature"]):
                for cuedleveli, cuedlevel in enumerate(["cued", "uncued"]):
                    for dayi, day in enumerate(["pre-training", "post-training"]):

                        SubID.append(traininggroupstring + str(sub_val))
                        SubIDval.append(sub_count + traininggroupcount * 37)
                        traininggroup.append(traininggroupstring)

                        testday.append(day)
                        cuetype.append(cuetypeval)
                        attentionstate.append(cuedlevel)

                        runlength.append(data[cuedleveli, cuetypei, dayi])

    data = {'SubID': SubID, 'SubIDval': SubIDval, 'Training Group': traininggroup,
            'Test Day': testday, 'Cuetype': cuetype,
            'CuedUncued': attentionstate, 'Run Length': runlength}
    df_runs = pd.DataFrame(data)


    # Runs
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    colors = [settings.lightteal, settings.darkteal]
    df_trainingplot =df_runs.loc[df_runs['Test Day'].isin(["pre-training"]) & df_runs['Cuetype'].isin(["Space"]), ]
    sns.violinplot(x='Training Group', y="Run Length", hue='CuedUncued', data=df_trainingplot, palette=sns.color_palette(colors), style="ticks",
                   ax=ax, inner="box", alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # titlestring = 'RSA training distance training effect Neurofeedback participants'
    # plt.suptitle(titlestring)
    # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

    df_runstraining = df_runs.loc[df_runs['Test Day'].isin(["pre-training"]) , ].reset_index().copy()
    df_runstraining['Run Length train'] = df_runs.loc[df_runs['Test Day'].isin(["post-training"]),'Run Length'].reset_index()['Run Length'] - df_runs.loc[df_runs['Test Day'].isin(["pre-training"]),'Run Length'].reset_index()['Run Length']
    df_runstraining = df_runstraining.loc[df_runstraining['CuedUncued']=="cued",]
    df_runstraining.groupby(['Training Group', 'Cuetype']).mean()

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    colors = [settings.lightteal, settings.darkteal]
    sns.violinplot(x='Training Group', y="Run Length train", hue='Cuetype', data=df_runstraining, palette=sns.color_palette(colors), style="ticks",
                   ax=ax, inner="box", alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    import scipy.stats as stats
    cuetype = "Feature"
    cat1 = df_runstraining.loc[np.logical_and(df_runstraining['Training Group'] == 'Space', df_runstraining['Cuetype']==cuetype), ]
    cat2 = df_runstraining.loc[np.logical_and(df_runstraining['Training Group'] == 'Feature', df_runstraining['Cuetype']==cuetype), ]
    cat12 = df_runstraining.loc[np.logical_and(df_runstraining['Training Group'].isin(['Feature', 'Space']), df_runstraining['Cuetype']==cuetype), ]
    cat3 = df_runstraining.loc[np.logical_and(df_runstraining['Training Group'] == 'Sham', df_runstraining['Cuetype']==cuetype), ]

    measure = "Run Length train"
    stats.ttest_ind(cat1[measure], cat3[measure])
    stats.ttest_ind(cat2[measure], cat3[measure])
    stats.ttest_ind(cat12[measure], cat3[measure])

    stats.ttest_1samp(cat1[measure], 0)
    stats.ttest_1samp(cat2[measure], 0)
    stats.ttest_1samp(cat3[measure], 0)


