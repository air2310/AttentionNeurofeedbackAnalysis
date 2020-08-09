# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift
import Analysis_Code.helperfunctions_ATTNNF as helper

# setup generic settings
attntrained = 1 # ["Space", "Feature"]
settings = helper.SetupMetaData(attntrained)

print("Analysing Data for condition train: " + settings.string_attntrained[settings.attntrained])

# iterate through subjects
for sub_count, sub_val in enumerate(settings.subsIDX):

    # decide whether to analyse EEG Pre Vs. Post Training
    analyseEEGprepost = True
    if (analyseEEGprepost):
        # get settings specific to this analysis
        settings = settings.get_settings_EEG_prepost()

        # get timing settings
        # timelimits_data, timepoints, frequencypoints, zeropoint = get_timing_variables(settings.timelimits,settings.samplingfreq)
        timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_zeropad,settings.samplingfreq)


        # preallocate
        num_epochs = settings.num_trials/4
        epochs_days = np.empty(( settings.num_electrodes, len(timepoints_zp) + 1, settings.num_levels, settings.num_attnstates,
                              settings.num_days))

        epochs_days[:] = np.nan


        # iterate through test days to get data
        for day_count, day_val in enumerate(settings.daysuse):
            # get file names
            bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
            print(bids.casestring)

            # get EEG data
            raw, events = helper.get_eeg_data(bids)

            # Filter Data
            raw.filter(l_freq=1, h_freq=45, h_trans_bandwidth=0.1)

            # Epoch to events of interest
            event_id = {'Space/Left_diag': 121, 'Space/Right_diag': 122,
                        'Feat/Black': 123, 'Feat/White': 124} # will be different triggers for training days

            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=settings.timelimits_zeropad[0], tmax=settings.timelimits_zeropad[1],
                                baseline=(0, 1 / settings.samplingfreq), picks=np.arange(settings.num_electrodes),
                                reject=dict(eeg=400), detrend=1)

            # drop bad channels
            epochs.drop_bad()
            epochs.plot_drop_log()
            # epochs2 = epochs.equalize_event_counts(event_id, method='mintime')

            # for fft - get data for each  condition
            erps_days[:, :, 0, 0, day_count] = np.squeeze(np.mean(epochs['Space/Left_diag'].get_data(), axis=0))
            erps_days[:, :, 0, 1, day_count] = np.squeeze(np.mean(epochs['Space/Right_diag'].get_data(), axis=0))
            erps_days[:, :, 1, 0, day_count] = np.squeeze(np.mean(epochs['Feat/Black'].get_data(), axis=0))
            erps_days[:, :, 1, 1, day_count] = np.squeeze(np.mean(epochs['Feat/White'].get_data(), axis=0))

            # ERPs and wavelets
            # erp_space_right = epochs['Space/Left_diag'].average()
            # erp_space_right = epochs['Space/Right_diag'].average()
            # erp_feat_black = epochs['Feat/Black'].average()
            # erp_feat_white = epochs['Feat/White'].average()
            #
            # erp_space_left.plot()
            # erp_space_right.plot()
            # erp_feat_black.plot()
            # erp_feat_white.plot()
            # get wavelet data
            # freqs = np.reshape(settings.hz_attn, -1)
            # ncycles = freqs
            # wave_space_left = mne.time_frequency.tfr_morlet(epochs['Space/Left_diag'].average(), freqs, ncycles, return_itc=False)
            # wave_space_left.plot(picks=np.arange(9),vmin=-500, vmax=500, cmap='viridis')


        # for fft - zeropad
        zerotimes = np.where(
            np.logical_or(epochs.times < settings.timelimits[0], epochs.times > settings.timelimits[1]))
        erps_days[:, zerotimes, :, :, :] = 0

        # for fft - fft
        fftdat = np.abs(fft(erps_days, axis=1)) / len(epochs.times)

        # for fft - plot data
        # to do: make subplots for days, set titles and x and y lablels, set ylim to be comparable, get rid of zero line, save
        plt.figure()
        freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
        chanmeanfft = np.mean(fftdat, axis=0)
        plt.plot(freq, chanmeanfft[ :, 0, 0, 0].T,'-', label='Space/Left_diag') # 'Space/Left_diag'
        plt.plot(freq, chanmeanfft[ :, 0, 1, 0].T,'-' , label='Space/Right_diag') # 'Space/Right_diag'
        plt.plot(freq, chanmeanfft[ :, 1, 0, 0].T,'-' , label='Feat/Black') # 'Feat/Black'
        plt.plot(freq, chanmeanfft[ :, 1, 1, 0].T,'-' , label='Feat/White') # 'Feat/White'
        plt.legend()
        plt.title('day 1')
        plt.xlim(2, 20)

        # get indices for frequencies of interest
        hz_attn_index = np.empty((settings.num_spaces, settings.num_features))
        for space_count, space in enumerate(['Left_diag', 'Right_diag']):
            for feat_count, feat in enumerate(['Black', 'White']):
                hz_attn_index[space_count, feat_count] = np.argmin(np.abs(freq - settings.hz_attn[space_count, feat_count]))


        # get ssveps for space condition, sorted to represent attended vs. unattended
        cuetype = 0  # space
        left_diag, right_diag = 0, 1

        spaceSSVEPs = np.empty((settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_levels))
        for level_count, level in enumerate(['Left_diag', 'Right_diag']):
            if (level == 'Left_diag'):
                attendedSSVEPs = np.mean(fftdat[:, hz_attn_index[left_diag, :].astype(int), cuetype, level_count, :], axis=1)  # average across left_diag frequencies at both feature positions
                unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[right_diag, :].astype(int), cuetype, level_count, :], axis=1)  # average across right_diag frequencies at both feature positions

            if (level == 'Right_diag'):
                attendedSSVEPs = np.mean(fftdat[:, hz_attn_index[right_diag, :].astype(int), cuetype, level_count, :], axis=1)  # average across right_diag frequencies at both feature positions
                unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[left_diag, :].astype(int), cuetype, level_count, :], axis=1)  # average across left_diag frequencies at both feature positions

            spaceSSVEPs[:, :, 0, level_count] = attendedSSVEPs
            spaceSSVEPs[:, :, 1, level_count] = unattendedSSVEPs


        # get ssveps for feature condition, sorted to represent attended vs. unattended
        cuetype = 1 # feature
        black, white = 0, 1
        featureSSVEPs = np.empty((settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_levels))
        for level_count, level in enumerate(['Black', 'White']):
            print(level_count, level)
            if (level == 'Black'):
                attendedSSVEPs =   np.mean(fftdat[:, hz_attn_index[:, black].astype(int), cuetype, level_count, :], axis = 1) # average across black frequencies at both spatial positions
                unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, white].astype(int), cuetype, level_count, :], axis = 1) # average across white frequencies at both spatial positions

            if (level == 'White'):
                attendedSSVEPs =   np.mean(fftdat[:, hz_attn_index[:, white].astype(int), cuetype, level_count, :], axis = 1) # average across white frequencies at both spatial positions
                unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, black].astype(int), cuetype, level_count, :], axis = 1) # average across black frequencies at both spatial positions

            featureSSVEPs[:, :, 0, level_count] = attendedSSVEPs
            featureSSVEPs[:, :, 1, level_count] = unattendedSSVEPs


        # average across cue types and store the SSVEPs alltogether for plotting and further analysis
        SSVEPs_prepost = np.empty((settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_attnstates))
        SSVEPs_prepost[:, :, :, 0] = np.mean(spaceSSVEPs, axis=3)
        SSVEPs_prepost[:, :, :, 1] = np.mean(featureSSVEPs, axis=3)

        # plot topos
        # TODO* add topoplotting
        # TODO : fix plots all over the place
        # TODO: Multiple errors can be wrapped inside an exception.

        # get best electrodes to use
        BEST = np.empty((settings.num_best, settings.num_days, settings.num_attnstates))
        SSVEPs_prepost_mean = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates))
        for day_count, day_val in enumerate(settings.daysuse):
            for attn_count, attn_val in enumerate(settings.string_attntrained):
                tmp = np.mean(SSVEPs_prepost[:,day_count,:,attn_count], axis=1)
                BEST[:,day_count, attn_count] = tmp.argsort()[-settings.num_best:]

                SSVEPs_prepost_mean[:,day_count, attn_count] = np.mean(SSVEPs_prepost[BEST[:,day_count, attn_count].astype(int), day_count, :, attn_count], axis=0)

        # get mean SSVEPs


        # plot results
        fig, (ax1, ax2) = plt.subplots(1, 2)

        labels = settings.string_attd_unattd
        x = np.arange(len(labels))
        width = 0.35

        for attn in np.arange(settings.num_attnstates):
            if (attn==0):axuse = ax1
            if (attn==1):axuse = ax2

            axuse.bar(x - width/2, SSVEPs_prepost_mean[:,0, attn], width, label=settings.string_prepost[0]) # Pretrain
            axuse.bar(x + width / 2, SSVEPs_prepost_mean[:,1, attn], width, label=settings.string_prepost[1]) # Posttrain

            # Add some text for labels, title and custom x-axis tick labels, etc.
            axuse.set_ylabel('SSVEP amp (µV)')
            axuse.set_title(settings.string_cuetype[attn])
            axuse.set_xticks(x)
            axuse.set_xticklabels(labels)
            axuse.legend()

        # next step - compute differences and plot

        fig = plt.figure()
        labels = settings.string_prepost
        x = np.arange(len(labels))
        width = 0.35

        attn = 0
        datplot = SSVEPs_prepost_mean[0,:, attn] - SSVEPs_prepost_mean[1,:, attn]
        plt.bar(x - width/2, datplot, width, label=settings.string_attntrained[attn]) # 'space'

        attn = 1
        datplot = SSVEPs_prepost_mean[0, :, attn] - SSVEPs_prepost_mean[1, :, attn]
        plt.bar(x + width / 2, datplot, width, label=settings.string_attntrained[attn])  # 'feature'

        plt.ylabel('Delta SSVEP amp (µV)')
        plt.title(settings.string_cuetype[attn])
        plt.xticks(x)
        # plt.xticklabels(labels)
        plt.legend()

        # when we return - check single trial SSVEP amplitudes. figure out if this script is wrong or if the matlab script is.