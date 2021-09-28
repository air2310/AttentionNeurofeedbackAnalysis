import numpy as np
from pathlib import Path
import mne
import helperfunctions_ATTNNF as helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def analyseEEGprepost(settings, sub_val):
    print('analysing SSVEP amplitudes pre Vs. post training')

    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # get timing settings
    # timelimits_data, timepoints, frequencypoints, zeropoint = get_timing_variables(settings.timelimits,settings.samplingfreq)
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(
        settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate
    num_epochs = settings.num_trials / settings.num_conditions
    epochs_days = np.empty(
        (int(num_epochs), settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels,
         settings.num_days))
    epochs_days[:] = np.nan

    # iterate through test days to get data
    for day_count, day_val in enumerate(settings.daysuse):
        # TODO: save daily plots for events, drop logs, ERPs and FFTs
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
        
        epochs_days[0:len(epochs['Space/Left_diag']), :, :, 0, 0, day_count] = epochs['Space/Left_diag'].get_data()
        epochs_days[0:len(epochs['Space/Right_diag']), :, :, 0, 1, day_count] = epochs['Space/Right_diag'].get_data()
        epochs_days[0:len(epochs['Feat/Black']), :, :, 1, 0, day_count] = epochs['Feat/Black'].get_data()
        epochs_days[0:len(epochs['Feat/White']), :, :, 1, 1, day_count] = epochs['Feat/White'].get_data()

       # average across trials
    erps_days = np.squeeze(np.nanmean(epochs_days, axis=0))
    erps_days_wave = np.squeeze(np.nanmean(epochs_days, axis=0))
    timepoints_zp = epochs.times

    # # Get SSVEPs
    fftdat, fftdat_epochs, freq = getSSVEPs(erps_days, epochs_days, epochs, settings, bids)
    fftdat_epochs = np.nanmean(fftdat_epochs,axis=0)  # average across trials to get the same shape for single trial SSVEPs

    # get signal to noise
    fftdat_snr = getfft_sigtonoise(settings, epochs, fftdat, freq)
    fftdat_snr_epochs = getfft_sigtonoise(settings, epochs, fftdat_epochs, freq)

    # get ssvep amplitudes
    SSVEPs_prepost, SSVEPs_prepost_channelmean, BEST = getSSVEPS_conditions(settings, fftdat, freq)  # trial average
    SSVEPs_prepost_epochs, SSVEPs_prepost_channelmean_epochs, BEST_epochs = getSSVEPS_conditions(settings, fftdat_epochs, freq)  # single trial

    # get ssvep amplitudes SNR
    SSVEPs_prepost_snr, SSVEPs_prepost_channelmean_snr, BEST_snr = getSSVEPS_conditions(settings, fftdat_snr,
                                                                                               freq)  # trial average
    SSVEPs_prepost_epochs_snr, SSVEPs_prepost_channelmean_epochs_snr, BEST_epochs_snr = getSSVEPS_conditions(
        settings, fftdat_snr_epochs, freq)  # single trial


    # Topoplot SSVEPs
    ERPstring = 'ERP'
    topoinfo = topoplot_SSVEPs(raw, SSVEPs_prepost, ERPstring, settings, bids)

    ERPstring = 'Single Trial'
    topoinfo = topoplot_SSVEPs(raw, SSVEPs_prepost_epochs, ERPstring, settings, bids)

    # get wavelets
    wavelets_prepost = get_wavelets_prepost(erps_days_wave, settings, epochs, BEST, bids)

    # Plot SSVEP results
    ERPstring = 'ERP'
    plotResultsPrePost_subjects(SSVEPs_prepost_channelmean, settings, ERPstring, bids)

    ERPstring = 'Single Trial'
    plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs, settings, ERPstring, bids)

    # ERPstring = 'ERP SNR'
    # plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_snr, settings, ERPstring, bids)
    #
    # ERPstring = 'Single Trial SNR'
    # plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs_snr, settings, ERPstring, bids)
    #


    np.savez(bids.direct_results / Path(bids.substring + "EEG_pre_post_results"),
             SSVEPs_prepost_channelmean_epochs_snr=SSVEPs_prepost_channelmean_epochs_snr,
             SSVEPs_prepost_channelmean_snr=SSVEPs_prepost_channelmean_snr,
             SSVEPs_prepost_channelmean_epochs=SSVEPs_prepost_channelmean_epochs,
             SSVEPs_prepost_channelmean=SSVEPs_prepost_channelmean,
             SSVEPs_prepost_epochs=SSVEPs_prepost_epochs,
             SSVEPs_prepost=SSVEPs_prepost,
             wavelets_prepost=wavelets_prepost,
             timepoints_zp=timepoints_zp,
             erps_days_wave=erps_days_wave,
             fftdat=fftdat, fftdat_epochs=fftdat_epochs, freq=freq, topoinfo=topoinfo)


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


def getfft_sigtonoise(settings, epochs, fftdat, freq):

    # get SNR fft
    numsnr = 10
    extradatapoints = (numsnr*2 + 1)
    numdatpoints = 7200
    snrtmp = np.empty((settings.num_electrodes, numdatpoints+extradatapoints , settings.num_attnstates, settings.num_levels, settings.num_days, extradatapoints ))
    snrtmp[:] = np.NAN

    for i in np.arange(extradatapoints):
        if (i != numsnr):
            snrtmp[:, i:-(extradatapoints - i), :, :, :, i] = fftdat

    snr = fftdat / np.nanmean(snrtmp[:,numsnr:-(extradatapoints - numsnr),:,:,:,:], axis=5)

    return snr


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


def get_wavelets_prepost(erps_days, settings, epochs, BEST, bids):
    import mne
    import matplotlib.pyplot as plt
    from pathlib import Path

    erps_days_wave = erps_days.transpose(2, 3, 4, 0, 1 )  # [cuetype, level, day, chans,time,]

    # Get wavelets for each frequency
    wavelets = np.empty((settings.num_attnstates, settings.num_levels, len(epochs.times), settings.num_spaces, settings.num_features, settings.num_days))
    for space_count, space in enumerate(['Left_diag', 'Right_diag']):
        for feat_count, feat in enumerate(['Black', 'White']):
            for day_count, day_val in enumerate(settings.daysuse):
                bestuse = BEST[:, space_count, feat_count, day_count].astype(int)
                hzuse = settings.hz_attn[space_count, feat_count]
                datuse = np.mean(erps_days_wave[:, :, day_count, bestuse, :], axis=2)
                wavelets[:, :, :, space_count, feat_count, day_count] = np.squeeze(mne.time_frequency.tfr_array_morlet(datuse, settings.samplingfreq, freqs=[hzuse],n_cycles=[hzuse], output='power'))


    # Space wavelets
    cuetype = 0  # space
    left_diag, right_diag = 0, 1

    # spaceSSVEPs = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_levels))
    spacewavelets = np.empty((len(epochs.times), settings.num_days, settings.num_attd_unattd, settings.num_levels))
    for level_count, level in enumerate(['Left_diag', 'Right_diag']):  # cycle through space trials on which the left and right diag were cued
        if (level == 'Left_diag'):  # when left diag cued
            attendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, left_diag, :, :], 1)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, right_diag, :, :], 1) # average across right_diag frequencies at both features (black, white)

        if (level == 'Right_diag'):  # when right diag cued
            attendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, right_diag, :, :], 1)  # average across right_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, left_diag, :, :], 1)  # average across left_diag frequencies at both features (black, white)

        spacewavelets[:, :, 0, level_count] = attendedSSVEPs
        spacewavelets[:, :, 1, level_count] = unattendedSSVEPs


    # feature wavelets
    cuetype = 1  # space
    black, white = 0, 1

    featurewavelets = np.empty((len(epochs.times), settings.num_days, settings.num_attd_unattd, settings.num_levels))
    for level_count, level in enumerate(['Black', 'White']):  # cycle through space trials on which the black and white were cued
        if (level == 'Black'):  # when left diag cued
            attendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, :, black, :],
                                     1)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, :, white, :],
                                       1)  # average across right_diag frequencies at both features (black, white)

        if (level == 'White'):  # when right diag cued
            attendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, :, white, :],
                                     1)  # average across right_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(wavelets[cuetype, level_count, :, :, black, :],
                                       1)  # average across left_diag frequencies at both features (black, white)

        featurewavelets[:, :, 0, level_count] = attendedSSVEPs
        featurewavelets[:, :, 1, level_count] = unattendedSSVEPs

    # average across cue types and store the SSVEPs alltogether for plotting and further analysis
    wavelets_prepost = np.empty((len(epochs.times), settings.num_days, settings.num_attd_unattd, settings.num_attnstates))
    wavelets_prepost[:, :, :, 0] = np.mean(spacewavelets, axis=3)
    wavelets_prepost[:, :, :, 1] = np.mean(featurewavelets, axis=3)

    # plot wavelet data
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 15))
    for attn in np.arange(settings.num_attnstates):
        for dayuse in np.arange(settings.num_days):
            if (dayuse == 0): axuse = ax1[attn]
            if (dayuse == 1): axuse = ax2[attn]
            if (attn == 0):
                axuse.plot(epochs.times, wavelets_prepost[:, dayuse, 0, attn], color=settings.medteal,
                           label=settings.string_attd_unattd[0])
                axuse.plot(epochs.times, wavelets_prepost[:, dayuse, 1, attn], color=settings.lightteal,
                           label=settings.string_attd_unattd[1])
            else:
                axuse.plot(epochs.times, wavelets_prepost[:, dayuse, 0, attn], color=settings.orange,
                           label=settings.string_attd_unattd[0])
                axuse.plot(epochs.times, wavelets_prepost[:, dayuse, 1, attn], color=settings.yellow,
                           label=settings.string_attd_unattd[1])
            axuse.set_xlim(-1, 6)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn] + ' ' + settings.string_prepost[dayuse])

    titlestring = bids.substring + ' wavelets pre-post'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    return wavelets_prepost


def topoplot_SSVEPs(raw, SSVEPs, ERPstring, settings, bids):
    import matplotlib.pyplot as plt
    from pathlib import Path

    # define expanded montage
    montage = {'Iz': [0, -110, -40],
               'Oz': [0, -105, -15],
               'POz': [0, -100, 15],
               'O1': [-40, -106, -15],
               'O2': [40, -106, -15],
               'PO3': [-35, -101, 10],
               'PO4': [35, -101, 10],
               'PO7': [-70, -110, 0],
               'PO8': [70, -110, 0],
               'Pz': [0, -95, 45],
               'P9': [-120, -110, 15],
               'P10': [120, -110, 15]}

    montageuse = mne.channels.make_dig_montage(ch_pos=montage, lpa=[-82.5, -19.2, -46], nasion=[0, 83.2, -38.3],
                                               rpa=[82.2, -19.2,
                                                    -46])  # based on mne help file on setting 10-20 montage

    # make new fake channels to add to dat so we can expand info out and add our extra (empty) channels for plotting
    tmp = raw.copy().pick_types(eeg=True, exclude=['TRIG'])
    tmp.rename_channels({"Iz": "Pz"})
    tmp.rename_channels({"Oz": "P9"})
    tmp.rename_channels({"POz": "P10"})
    tmp.pick_channels([tmp.ch_names[pick] for pick in np.arange(3)])

    # Add the empty channels to our dopodat (just to get the info) and apply the expanded montage
    topodat = raw.copy().pick_types(eeg=True, exclude=['TRIG'])
    topodat.add_channels([tmp])

    topodat.info.set_montage(montageuse)

    # attntype, day, attd = 0, 0, 0
    # plot topomap
    vmin, vmax = np.min(SSVEPs[:]), np.max(SSVEPs[:])  # get limits

    fig, (ax1, ax2) = plt.subplots(2, 4, figsize=(15, 6))
    count = -1
    for attntype in np.arange(2):
        for day in np.arange(2):
            count += 1
            for attd in np.arange(2):
                if (attd == 0): axuse = ax1
                if (attd == 1): axuse = ax2

                plt.axes(axuse[count])
                # np.empty((settings.num_electrodes, settings.num_attd_unattd, settings.num_days, settings.num_attnstates))
                dataplot = SSVEPs[:, attd, day, attntype]  # chans, # daycount, # attd unattd, # space/feat
                dataplot = np.append(dataplot, [0, 0, 0])

                im = mne.viz.plot_topomap(dataplot, topodat.info, cmap="viridis", show_names=False,
                                     names=list(('Iz', 'Oz', 'POz', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8', 'Pz',
                                                 'P9', 'P10')), vmin=vmin, vmax=vmax, contours = 0)

                plt.title(settings.string_attntrained[attntype] + " " + settings.string_attd_unattd[attd] + " " +
                          settings.string_prepost[day])

                # plt.colorbar(plt.cm.ScalarMappable(cmap=im[0].cmap))
                plt.colorbar(im[0], shrink=0.5)


    titlestring = bids.substring + ' ' + ERPstring + ' Topoplots pre-post'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    return topodat.info


def plotResultsPrePost_subjects(SSVEPs_prepost_mean, settings, ERPstring, bids):
    import matplotlib.pyplot as plt
    from pathlib import Path

    # plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    labels = settings.string_attd_unattd
    x = np.arange(len(labels))
    width = 0.35

    for attn in np.arange(settings.num_attnstates):
        if (attn == 0): axuse = ax1
        if (attn == 1): axuse = ax2

        axuse.bar(x - width / 2, SSVEPs_prepost_mean[:, 0, attn], width, label=settings.string_prepost[0],
                  facecolor=settings.lightteal)  # Pretrain
        axuse.bar(x + width / 2, SSVEPs_prepost_mean[:, 1, attn], width, label=settings.string_prepost[1],
                  facecolor=settings.medteal)  # Posttrain

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axuse.set_ylabel('SSVEP amp (µV)')
        axuse.set_title(settings.string_cuetype[attn])
        axuse.set_xticks(x)
        axuse.set_xticklabels(labels)
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = bids.substring + ' ' + ERPstring + ' SSVEPs pre-post'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # next step - compute differences and plot
    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_cuetype
    x = np.arange(len(labels))
    width = 0.35

    day = 0
    datplot = SSVEPs_prepost_mean[0, day, :] - SSVEPs_prepost_mean[1, day, :]
    plt.bar(x - width / 2, datplot, width, label=settings.string_prepost[day], facecolor=settings.yellow)  # 'space'

    day = 1
    datplot = SSVEPs_prepost_mean[0, day, :] - SSVEPs_prepost_mean[1, day, :]
    plt.bar(x + width / 2, datplot, width, label=settings.string_prepost[day], facecolor=settings.orange)  # 'feature'

    plt.ylabel('Delta SSVEP amp (µV)')
    plt.title(settings.string_cuetype[attn])
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = bids.substring + ' ' + ERPstring + ' SSVEP selectivity pre-post'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')


def topoplot_SSVEPs_group(raw, SSVEPs, ERPstring, settings, bids):
    import matplotlib.pyplot as plt
    from pathlib import Path
    SSVEPs_mean = np.nanmean(SSVEPs, axis=4)
    # define expanded montage
    montage = {'Iz': [0, -140, -40],
               'Oz': [0, -130, -15],
               'POz': [0, -120, 15],
               'O1': [-45, -125, -15],
               'O2': [45, -125, -15],
               'PO3': [-45, -120, 10],
               'PO4': [45, -120, 10],
               'PO7': [-70, -130, -10],
               'PO8': [70, -130, -10],
               'Pz': [0, -95, 45],
               'P9': [-80, -110, 15],
               'P10': [80, -110, 15]}

    montageuse = mne.channels.make_dig_montage(ch_pos=montage, lpa=[-82.5, -19.2, -46], nasion=[0, 83.2, -38.3],
                                               rpa=[82.2, -19.2,
                                                    -46])  # based on mne help file on setting 10-20 montage

    # make new fake channels to add to dat so we can expand info out and add our extra (empty) channels for plotting
    tmp = raw.copy().pick_types(eeg=True, exclude=['TRIG'])
    tmp.rename_channels({"Iz": "Pz"})
    tmp.rename_channels({"Oz": "P9"})
    tmp.rename_channels({"POz": "P10"})
    tmp.pick_channels([tmp.ch_names[pick] for pick in np.arange(3)])

    # Add the empty channels to our dopodat (just to get the info) and apply the expanded montage
    topodat = raw.copy().pick_types(eeg=True, exclude=['TRIG'])
    topodat.add_channels([tmp])

    topodat.info.set_montage(montageuse)

    # attntype, day, attd = 0, 0, 0
    # plot topomap
    vmin, vmax = np.min(SSVEPs_mean[:]), np.max(SSVEPs_mean[:])  # get limits

    fig, (ax1, ax2) = plt.subplots(2, 4, figsize=(15, 6))
    count = -1
    for attntype in np.arange(2):
        for day in np.arange(2):
            count += 1
            for attd in np.arange(2):
                if (attd == 0): axuse = ax1
                if (attd == 1): axuse = ax2

                plt.axes(axuse[count])
                dataplot = SSVEPs_mean[:,  attd, day, attntype]  # chans, # daycount, # attd unattd, # space/feat
                dataplot = np.append(dataplot, [0, 0, 0])

                im = mne.viz.plot_topomap(dataplot, topodat.info, cmap="viridis", show_names=False,
                                          names=list(('Iz', 'Oz', 'POz', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8', 'Pz',
                                                      'P9', 'P10')), vmin=vmin, vmax=vmax, contours=0)

                plt.title(settings.string_cuetype[attntype] + " " + settings.string_attd_unattd[attd] + " " +
                          settings.string_prepost[day])

                # plt.colorbar(plt.cm.ScalarMappable(cmap=im[0].cmap))
                plt.colorbar(im[0], shrink=0.5)

    titlestring = ERPstring + ' Topoplots pre-post'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    ####### Difference Plot ( Attd - unattd)

    SSVEPs_Select = SSVEPs_mean[:, 0, :, :] - SSVEPs_mean[:, 1, :, :]  # attntype, day, attd

    fig, ax = plt.subplots(2, 2, figsize=(8, 8))
    count = -1
    for attntype in np.arange(2):
        vmin, vmax = 0, np.max(np.abs(SSVEPs_Select[:, :, attntype]))  # get limits
        for day in np.arange(2):

            plt.axes(ax[attntype, day])
            dataplot =SSVEPs_Select[:, day, attntype]  # chans, # daycount, # attd unattd, # space/feat
            dataplot = np.append(dataplot, [0, 0, 0])

            im = mne.viz.plot_topomap(dataplot, topodat.info, cmap="viridis", show_names=False,
                                      names=list(('Iz', 'Oz', 'POz', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8', 'Pz',
                                                  'P9', 'P10')), vmin=vmin, vmax=vmax, contours=0)

            plt.title(settings.string_cuetype[attntype] + " "  + settings.string_prepost[day])

            # plt.colorbar(plt.cm.ScalarMappable(cmap=im[0].cmap))
            plt.colorbar(im[0], shrink=0.5)

    titlestring = ERPstring + ' Topoplots pre-post Selectivity'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    ####### Difference Plot ( Attd - unattd)(Day4 - Day1)

    SSVEPs_Select_day = SSVEPs_Select[:, 1, :] - SSVEPs_Select[:, 0, :]  #  day, attd

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    vmin, vmax =  -0.035, 0.035 #-np.max(np.abs(SSVEPs_Select_day[:])), np.max(np.abs(SSVEPs_Select_day[:]))  # get limits

    for attntype in np.arange(2):

        plt.axes(ax[attntype])
        dataplot = SSVEPs_Select_day[:, attntype]  # chans, # daycount, # attd unattd, # space/feat
        dataplot = np.append(dataplot, [0, 0, 0])

        im = mne.viz.plot_topomap(dataplot, topodat.info, cmap="viridis", show_names=False,
                                  names=list(('Iz', 'Oz', 'POz', 'O1', 'O2', 'PO3', 'PO4', 'PO7', 'PO8', 'Pz',
                                              'P9', 'P10')), vmin=vmin, vmax=vmax, contours=0)

        plt.title(settings.string_cuetype[attntype] )

        # plt.colorbar(plt.cm.ScalarMappable(cmap=im[0].cmap))
        plt.colorbar(im[0], shrink=0.5)

    titlestring = ERPstring + ' Topoplots selectivity training efct'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


def plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring, settings, freq):
    import matplotlib.pyplot as plt
    from pathlib import Path
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    chanmeanfft = np.mean(fftdat_ave, axis=0)

    for day_count in np.arange(2):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        hz_attn = settings.hz_attn  # ['\B \W'], ['/B /W']
        axuse.axvline(hz_attn[0, 0], 0, 1, linestyle='--', color='k', alpha=0.2)  # black/Left_diag
        axuse.annotate("Black-leftdiag", (hz_attn[0, 0], 0.3))
        axuse.axvline(hz_attn[0, 1], 0, 1, linestyle='--', color='k', alpha=0.2)  # black/Right_diag
        axuse.annotate("Black-rightdiag", (hz_attn[0, 1], 0.3))
        axuse.axvline(hz_attn[1, 0], 0, 1, linestyle='--', color='k', alpha=0.2)  # white/Left_diag
        axuse.annotate("white-leftdiag", (hz_attn[1, 0], 0.3))
        axuse.axvline(hz_attn[1, 1], 0, 1, linestyle='--', color='k', alpha=0.2)  # white/Right_diag
        axuse.annotate("white-rightdiag", (hz_attn[1, 1], 0.3))

        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Space/Left_diag',
                   color=settings.lightteal, alpha=0.5)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Space/Right_diag',
                   color=settings.medteal, alpha=0.5)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='Feat/Black',
                   color=settings.orange, alpha=0.5)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='Feat/White',
                   color=settings.yellow, alpha=0.5)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        if ERPstring == 'Single Trial':
            axuse.set_ylim(0, 0.8)
        else:
            axuse.set_ylim(0, .4)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = 'Group Mean '+ ERPstring +' FFT Spectrum' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # Plot grand average figure
    fig, axuse = plt.subplots(1, 1, figsize=(12, 6))
    meanfft = np.mean(np.mean(np.mean(fftdat_ave, axis=2), axis=2), axis=2)
    chanmeanfft = np.mean(meanfft, axis=0)

    hz_attn = settings.hz_attn  # ['\B \W'], ['/B /W']
    axuse.axvline(hz_attn[0, 0], 0, 1, linestyle='--', color='k', alpha=0.2)  # black/Left_diag
    axuse.annotate("Black-leftdiag", (hz_attn[0, 0], 0.3))
    axuse.axvline(hz_attn[0, 1], 0, 1, linestyle='--', color='k', alpha=0.2)  # black/Right_diag
    axuse.annotate("Black-rightdiag", (hz_attn[0, 1], 0.3))
    axuse.axvline(hz_attn[1, 0], 0, 1, linestyle='--', color='k', alpha=0.2)  # white/Left_diag
    axuse.annotate("white-leftdiag", (hz_attn[1, 0], 0.3))
    axuse.axvline(hz_attn[1, 1], 0, 1, linestyle='--', color='k', alpha=0.2)  # white/Right_diag
    axuse.annotate("white-rightdiag", (hz_attn[1, 1], 0.3))

    axuse.plot(freq, chanmeanfft, '-', color='k', alpha=0.5)  # 'Space/Left_diag'

    axuse.set_xlim(2, 18)
    if ERPstring == 'Single Trial':
        axuse.set_ylim(0, 0.8)
    else:
        axuse.set_ylim(0, .4)
    axuse.legend()
    axuse.set_frame_on(False)
    axuse.spines['top'].set_visible(False)
    axuse.spines['right'].set_visible(False)

    titlestring = 'Group Mean ' + ERPstring + ' FFT Spectrum grandmean' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group / Path(titlestring + '.eps'), format='eps')

    # plot freq tagging
    SSVEPS_freq = np.empty((9,4))
    SSVEPS_freq[:, 0] = meanfft[:, np.argmin(np.abs(freq - settings.hz_attn[1, 0]))]
    SSVEPS_freq[:, 1] = meanfft[:, np.argmin(np.abs(freq - settings.hz_attn[0, 0]))]
    SSVEPS_freq[:, 2] = meanfft[:, np.argmin(np.abs(freq - settings.hz_attn[0, 1]))]
    SSVEPS_freq[:, 3] = meanfft[:, np.argmin(np.abs(freq - settings.hz_attn[1, 1]))]
    Hz_strings = ['4.5', '8.0', '12.0', '14.4']

    import matplotlib
    my_cmap = plt.get_cmap("Blues")#.reversed()
    y = np.hstack((SSVEPS_freq[:, 0], SSVEPS_freq[:, 1], SSVEPS_freq[:, 2], SSVEPS_freq[:, 3]))
    norm = matplotlib.colors.Normalize(vmin=0, vmax=0.5)
    # rescale = lambda y: (y - np.min(y)) / (np.max(y) - np.min(y))

    fig, axuse = plt.subplots(1, 4, figsize=(12, 4))
    channels = ['Iz', 'Oz', 'POz', 'O1', 'O2','PO3','PO4','PO7','PO8']
    for ii in np.arange(4):
        axuse[ii].bar(channels, SSVEPS_freq[:, ii], color=my_cmap(norm(SSVEPS_freq[:, ii])))
        axuse[ii].set_title(Hz_strings[ii])
        axuse[ii].set_ylim([0, 0.5])
        axuse[ii].spines['top'].set_visible(False)
        axuse[ii].spines['right'].set_visible(False)

    titlestring = 'SSVEPs across channels ' + ERPstring + " " + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group / Path(titlestring + '.eps'), format='eps')


def plotGroupSSVEPsprepost(SSVEPs_prepost_group, bids, ERPstring, settings):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import helperfunctions_ATTNNF as helper

    M = np.mean(SSVEPs_prepost_group, axis=3)

    E = np.empty((settings.num_attnstates, settings.num_days, settings.num_attnstates))
    for attn in np.arange(settings.num_attnstates):
        for day in np.arange(settings.num_days):
            E[:,day, attn] = helper.within_subjects_error(SSVEPs_prepost_group[:,day,attn,:].T)

    # E = np.std(SSVEPs_prepost_group, axis=3) / settings.num_subs

    # plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    labels = settings.string_attd_unattd
    x = np.arange(len(labels))
    width = 0.35

    for attn in np.arange(settings.num_attnstates):
        if (attn == 0): axuse = ax1
        if (attn == 1): axuse = ax2

        axuse.bar(x - width / 2, M[:, 0, attn], yerr=E[:, 0, attn], width=width, label=settings.string_prepost[0],
                  facecolor=settings.lightteal)  # Pretrain
        axuse.bar(x + width / 2, M[:, 1, attn], yerr=E[:, 1, attn], width=width, label=settings.string_prepost[1],
                  facecolor=settings.medteal)  # Posttrain

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axuse.set_ylabel('SSVEP amp (µV)')
        axuse.set_ylim([np.round(np.min(M)*100)/100-0.05, np.round(np.max(M)*100)/100+0.05])
        axuse.set_title(settings.string_cuetype[attn])
        axuse.set_xticks(x)
        axuse.set_xticklabels(labels)
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = 'Group Mean ' + ERPstring + ' SSVEPs pre-post TRAIN ' + settings.string_attntrained[
        settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # compute SSVEP differences and plot
    diffdat = SSVEPs_prepost_group[0, :, :, :] - SSVEPs_prepost_group[1, :, :, :]
    M = np.mean(diffdat, axis=2)
    # E = np.std(diffdat, axis=2) / settings.num_subs

    E = np.empty(( settings.num_days, settings.num_attnstates))
    for day in np.arange(settings.num_days):
        E[day, :] = helper.within_subjects_error(diffdat[day,:,:].T)


    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_cuetype
    x = np.arange(len(labels))
    width = 0.35
    xpos = np.array((x - width / 2, x + width / 2))
    colors = np.array((settings.yellow, settings.orange))
    for day in np.arange(settings.num_days):
        ax.bar(xpos[day], M[day, :], yerr=E[day, :], width=width, label=settings.string_prepost[day],
               facecolor=colors[day])
        # ax.plot(xpos[day] + np.random.random((settings.num_subs, settings.num_attnstates)) * width * 0.5 - width * 0.5 / 2,
        #         diffdat[day, :, :].T, '.', alpha=0.3, color='k')

    ax.plot(xpos[:, 0], diffdat[:, 0, :], '-', alpha=0.3, color='k')
    ax.plot(xpos[:, 1], diffdat[:, 1, :], '-', alpha=0.3, color='k')

    # day = 1
    # plt.bar(x + width / 2, M[day,:], yerr=E[day,:], width=width, label=settings.string_prepost[day], facecolor=settings.orange)  # 'feature'

    plt.ylabel('Delta SSVEP amp (µV)')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = 'Group Mean ' + ERPstring + ' SSVEP selectivity pre-post ' + settings.string_attntrained[
        settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


def collateEEGprepost(settings):
    print('collating SSVEP amplitudes pre Vs. post training')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(
        settings.timelimits_zeropad, settings.samplingfreq)

    timelimits_data, timepoints, frequencypoints, zeropoint = helper.get_timing_variables(
        settings.timelimits, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_topodat_group = np.empty(
        (settings.num_electrodes, settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_prepost_group = np.empty(
        (settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    fftdat_group = np.empty((settings.num_electrodes, len(timepoints), settings.num_attnstates,
                             settings.num_levels, settings.num_days, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints), settings.num_attnstates,
                                    settings.num_levels, settings.num_days, num_subs))
    wavelets_prepost_group = np.empty(
        (len(timepoints_zp) + 1, settings.num_days, settings.num_attd_unattd, settings.num_attnstates, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):

        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_pre_post_results.npz"), allow_pickle=True)  #
        # saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

        # store results
        SSVEPs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean']
        SSVEPs_epochs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']
        SSVEPs_topodat_group[:, :, :, :, sub_count] = results['SSVEPs_prepost_epochs']
        fftdat_group[:, :, :, :, :, sub_count] = results['fftdat']
        fftdat_epochs_group[:, :, :, :, :, sub_count] = results['fftdat_epochs']
        wavelets_prepost_group[:, :, :, :, sub_count] = results['wavelets_prepost']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

        if sub_count == 1:
            # get EEG data
            raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count=0, day_val=1,settings=settings)

    np.savez(bids.direct_results_group / Path("EEGResults_prepost"),
             SSVEPs_prepost_group=SSVEPs_prepost_group,
             SSVEPs_epochs_prepost_group=SSVEPs_epochs_prepost_group,
             SSVEPs_topodat_group=SSVEPs_topodat_group,
             fftdat_group=fftdat_group,
             fftdat_epochs_group=fftdat_epochs_group,
             wavelets_prepost_group=wavelets_prepost_group,
             timepoints_use=timepoints_use,
             freq=freq)

    # plot grand average frequency spectrum
    fftdat_ave = np.mean(fftdat_group, axis=5)
    plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring='ERP', settings=settings, freq=freq)

    fftdat_epochs_ave = np.mean(fftdat_epochs_group, axis=5)
    plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    plotGroupSSVEPsprepost(SSVEPs_prepost_group, bids, ERPstring='ERP', settings=settings)
    plotGroupSSVEPsprepost(SSVEPs_epochs_prepost_group, bids, ERPstring='Single Trial', settings=settings)

    topoplot_SSVEPs_group(raw, SSVEPs_topodat_group, ERPstring='ERP', settings=settings, bids=bids)

    # plot wavelet results
    wavelets_prepost_group = wavelets_prepost_group / 12001
    wavelets_prepost_ave = np.mean(wavelets_prepost_group, axis=4)
    wavelets_prepost_std = np.std(wavelets_prepost_group, axis=4) / num_subs

    # plot wavelet data
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 15))
    for attn in np.arange(settings.num_attnstates):
        for dayuse in np.arange(settings.num_days):
            if (dayuse == 0): axuse = ax1[attn]
            if (dayuse == 1): axuse = ax2[attn]
            if (attn == 0):
                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 0, attn] - wavelets_prepost_std[:, dayuse, 0, attn],
                                   wavelets_prepost_ave[:, dayuse, 0, attn] + wavelets_prepost_std[:, dayuse, 0, attn],
                                   alpha=0.3, facecolor=settings.medteal)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 0, attn], color=settings.medteal,
                           label=settings.string_attd_unattd[0])

                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 1, attn] - wavelets_prepost_std[:, dayuse, 1, attn],
                                   wavelets_prepost_ave[:, dayuse, 1, attn] + wavelets_prepost_std[:, dayuse, 1, attn],
                                   alpha=0.3, facecolor=settings.medteal)
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
            axuse.set_ylim(0.01, 0.07)
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn] + ' ' + settings.string_prepost[dayuse])

    titlestring = 'Group Mean wavelets pre-post ' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot difference waves
    diffwave = wavelets_prepost_group[:, :, 0, :, :] - wavelets_prepost_group[:, :, 1, :, :]
    wavelets_prepost_ave = np.mean(diffwave, axis=3)
    wavelets_prepost_std = np.std(diffwave, axis=3) / num_subs

    # plot wavelet data
    fig, (ax1) = plt.subplots(1, 2, figsize=(10, 15))
    for attn in np.arange(settings.num_attnstates):
        axuse = ax1[attn]
        coluse = [settings.medteal, settings.darkteal]
        for dayuse in np.arange(settings.num_days):
            axuse.axvline(0, 0, 1000, linewidth=2, color='k')
            if attn == 0:

                axuse.fill_between(timepoints_use, wavelets_prepost_ave[:, dayuse, attn] - wavelets_prepost_std[:, dayuse, attn], wavelets_prepost_ave[:, dayuse, attn] + wavelets_prepost_std[:, dayuse, attn], alpha=0.3, facecolor=coluse[dayuse])
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, attn], color=coluse[dayuse], label=settings.string_prepost[dayuse])

            else:
                axuse.fill_between(timepoints_use, wavelets_prepost_ave[:, dayuse, attn] - wavelets_prepost_std[:, dayuse, attn], wavelets_prepost_ave[:, dayuse, attn] + wavelets_prepost_std[:, dayuse, attn], alpha=0.3, facecolor=coluse[dayuse])
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, attn], color=coluse[dayuse], label=settings.string_prepost[dayuse])

            axuse.set_xlim(-1, 6)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.set_ylim(-0.02, 0.05)
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn])

    titlestring = 'Group Mean wavelets pre-post diff' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    # save attentional selectivity for stats - single trial
    ssvep_selectivity_prepost = SSVEPs_epochs_prepost_group[0, :, :, :] - SSVEPs_epochs_prepost_group[1, :, :, :]
    tmp = np.reshape(ssvep_selectivity_prepost,
                     (4, settings.num_subs))  # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost_epochs.npy"), tmp)

    # save attentional selectivity for stats
    ssvep_selectivity_prepost = SSVEPs_prepost_group[0, :, :, :] - SSVEPs_prepost_group[1, :, :, :]
    tmp = np.reshape(ssvep_selectivity_prepost,
                     (4, settings.num_subs))  # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost.npy"), tmp)

    np.save(bids.direct_results_group / Path("group_ssvep_prepost.npy"), SSVEPs_epochs_prepost_group)


def collateEEGprepostcompare(settings):

    print('collating SSVEP amplitudes pre Vs. post training compareing Space Vs. Feat Training')

    # preallocate
    num_subs = np.zeros((settings.num_attnstates))

    substrings_all = []
    daystrings = []
    attnstrings = []
    attntaskstrings = []

    selectivity_compare = []
    SSVEPs_attd = []
    SSVEPs_unattd = []
    SSVEPs_var = []

    # cycle trough space and feature train groups
    # for attntrained in np.arange(settings.num_attnstates):  # cycle trough space and feature train groups
    for attntrained, attntrainedstr in enumerate(settings.string_attntrained):  # cycle trough space, feature and sham train groups
        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_EEG_prepost()

        # load results
        bids = helper.BIDS_FileNaming(0, settings, 1)
        results = np.load(bids.direct_results_group / Path("EEGResults_prepost.npz"), allow_pickle=True)  #

        SSVEPs_prepost_group = results['SSVEPs_prepost_group'] #results['SSVEPs_epochs_prepost_group']  #[attd,day,cuetype,sub]
        diffdat = SSVEPs_prepost_group[0, :, :, :] - SSVEPs_prepost_group[1, :, :, :]  # [day,cuetype,sub]
        attddat = SSVEPs_prepost_group[0, :, :, :]
        unattddat = SSVEPs_prepost_group[1, :, :, :]

        SSVEPs_topodat_group = results['SSVEPs_topodat_group'] #[Chans,attd,day,cuetype,sub]
        diffdattopo = SSVEPs_topodat_group[:, 0, :, :, :] - SSVEPs_topodat_group[:, 1, :, :, :]  # [day,cuetype,sub]
        attddattopo = SSVEPs_topodat_group[:, 0, :, :, :]
        unattddattopo = SSVEPs_topodat_group[:, 1, :, :, :]
        diffdatvar = np.mean(diffdattopo[0:2, :, :, :],axis = 0)

        # file names
        substrings = []
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            for testday, daystring in enumerate(settings.string_prepost):
                for cue, cuestring in enumerate(["Space", "Feature"]):
                    substrings_all.append(sub_val)
                    daystrings.append(daystring)
                    attntaskstrings.append(cuestring)
                    attnstrings.append(attntrainedstr)

                    # data
                    selectivity_compare.append(diffdat[testday, cue, sub_count])
                    SSVEPs_attd.append(attddat[testday, cue, sub_count])
                    SSVEPs_unattd.append(unattddat[testday, cue, sub_count])
                    SSVEPs_var.append(diffdatvar[testday, cue, sub_count])

    data = {'SubID': substrings_all, 'Testday': daystrings, 'Attention Type': attntaskstrings,
            'Attention Trained': attnstrings, 'Selectivity (ΔµV)': selectivity_compare,
            'SSVEPs_attd': SSVEPs_attd, 'SSVEPs_unattd': SSVEPs_unattd, 'SSVEPs_var': SSVEPs_var}
    df_selectivity = pd.DataFrame(data)

    # # lets run some stats with R - save it out
    df_selectivity.to_csv(bids.direct_results_group_compare / Path("motiondiscrim_SelectivityResults_ALL.csv"),
                         index=False)

    ################# SSVEP Amps ##################
    df_grouped = df_selectivity.groupby(["SubID", "Attention Type"]).mean().reset_index()

    attd = df_grouped[["SubID", "Attention Type", "SSVEPs_attd"]].copy()
    unattd = df_grouped[["SubID", "Attention Type", "SSVEPs_unattd"]].copy()
    attd["Cue"] = 'Attended'
    unattd["Cue"] = 'Unattended'
    attd = attd.rename(columns={"SSVEPs_attd": "SSVEPs"})
    unattd = unattd.rename(columns={"SSVEPs_unattd": "SSVEPs"})

    df_SSVEPs = pd.concat([attd, unattd], ignore_index=True)

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # SSVEP Grouped violinplot
    colors = [settings.lightteal, settings.medteal]
    for i in np.arange(2):
        datplot = df_SSVEPs[df_SSVEPs["Attention Type"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Cue", y="SSVEPs", data=datplot, color="0", ax=ax[i])
        sns.violinplot(x="Cue", y="SSVEPs", data=datplot, palette=sns.color_palette(colors), style="ticks",
                       ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        ax[i].set_ylim([0, 1])
        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task SSVEP Amplitudes by attention type'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

    ################# SELECTIVITY #################
    # plot results - maximum split

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = ["#F2B035", "#EC553A"]

    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y="Selectivity (ΔµV)", hue="Testday",
                   data=df_selectivity[df_selectivity["Attention Type"].isin(["Space"])],
                   palette=sns.color_palette(colors), ax=ax1, split=True, inner="stick")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title(settings.string_attntrained[0] + " Attention")
    ax1.set_ylim(-0.5, 1.2)

    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y="Selectivity (ΔµV)", hue="Testday",
                   data=df_selectivity[df_selectivity["Attention Type"].isin(["Feature"])],
                   palette=sns.color_palette(colors), ax=ax2, split=True, inner="stick")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title(settings.string_attntrained[1] + " Attention")
    ax2.set_ylim(-0.25, 0.65)

    titlestring = 'Attentional Selectivity PrePost Compare Training'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    ################# SELECTIVITY VAR #################
    # plot results - maximum split

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = ["#F2B035", "#EC553A"]

    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y="SSVEPs_var", hue="Testday",
                   data=df_selectivity[df_selectivity["Attention Type"].isin(["Space"])],
                   palette=sns.color_palette(colors), ax=ax1, split=True, inner="stick")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title(settings.string_attntrained[0] + " Attention")
    # ax1.set_ylim(-0.5, 1.2)

    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y="SSVEPs_var", hue="Testday",
                   data=df_selectivity[df_selectivity["Attention Type"].isin(["Feature"])],
                   palette=sns.color_palette(colors), ax=ax2, split=True, inner="stick")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title(settings.string_attntrained[1] + " Attention")
    # ax2.set_ylim(-0.25, 0.65)

    titlestring = 'Attentional Selectivity Variance PrePost Compare Training'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    ##########################################  plot day 1 Vs. Day 4 Results ##########################################
    #### Selectivity Data ####
    df_grouped = df_selectivity.groupby(["SubID", "Testday"]).mean().reset_index()

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal]

    sns.swarmplot(x="Testday", y="Selectivity (ΔµV)", data=df_grouped, color="0",
                  order=["pre-training", "post-training"])
    sns.violinplot(x="Testday", y="Selectivity (ΔµV)", data=df_grouped, palette=sns.color_palette(colors),
                   style="ticks",
                   ax=ax1, inner="box", alpha=0.6, order=["pre-training", "post-training"])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task SSVEP Selectivity by Day pre Vs. post'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

    ##########################################  Calculate training effects  ##########################################
    idx_d1 = df_selectivity["Testday"] == "pre-training"
    idx_d4 = df_selectivity["Testday"] == "post-training"

    tmpd4 = df_selectivity[idx_d4].reset_index()
    tmpd1 = df_selectivity[idx_d1].reset_index()

    df_SSVEPtraineffects = tmpd4[["Attention Trained", "Attention Type"]].copy()
    df_SSVEPtraineffects["∆ Selectivity"] = tmpd4['Selectivity (ΔµV)'] - tmpd1['Selectivity (ΔµV)']
    df_SSVEPtraineffects["∆ Selectivity Var"] = tmpd4['SSVEPs_var'] - tmpd1['SSVEPs_var']
    ##########################################  plot training effects against attention trained and attention type ##########################################
    # df_SSVEPtraineffects = df_SSVEPtraineffects.drop(index=[139, 150])
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange, settings.red]

    for i in np.arange(2):
        datplot = df_SSVEPtraineffects[df_SSVEPtraineffects["Attention Type"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Trained", y="∆ Selectivity", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Trained", y="∆ Selectivity", data=datplot, palette=sns.color_palette(colors),
                       style="ticks",
                       ax=ax[i], inner="box", alpha=0.6)

        sns.lineplot(x="Attention Trained", y="∆ Selectivity", data=datplot, ax=ax[i],markers=True, dashes=False, color="k", err_style="bars", ci=68)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        if i == 0:
            ax[i].set_ylim([-0.75, 0.75])
        else:
            ax[i].set_ylim([-0.2, 0.2])
        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task SSVEP Selectivity training effect by attention'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')


    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange, settings.red]

    for i in np.arange(2):
        datplot = df_SSVEPtraineffects[df_SSVEPtraineffects["Attention Type"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Trained", y="∆ Selectivity Var", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Trained", y="∆ Selectivity Var", data=datplot, palette=sns.color_palette(colors),
                       style="ticks",
                       ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        # if i == 0:
            # ax[i].set_ylim([-0.75, 0.75])
        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task SSVEP Selectivity Var training effect by attention'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

