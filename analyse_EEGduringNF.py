import numpy as np
from pathlib import Path
import mne
import helperfunctions_ATTNNF as helper
import matplotlib.pyplot as plt
import seaborn as sns

def analyseEEG_duringNF(settings, sub_val):
    settings = settings.get_settings_EEG_duringNF()

    # get timing settings
    # timelimits_data, timepoints, frequencypoints, zeropoint = get_timing_variables(settings.timelimits,settings.samplingfreq)
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(
        settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate
    num_epochs = settings.num_trials / settings.num_conditions
    epochs_days = np.empty(
        (int(num_epochs), settings.num_electrodes, len(timepoints_zp) + 1, settings.num_features, settings.num_spaces,
         settings.num_days))
    epochs_days[:] = np.nan

    # iterate through test days to get data
    for day_count, day_val in enumerate(settings.daysuse):
        # TODO: save daily plots for events, drop logs, ERPs and FFTs

        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # make sure file exists
        possiblefiles = []
        filesizes = []
        for filesfound in bids.direct_data_eeg.glob(bids.filename_eeg + "*.eeg"):
            filesizes.append(filesfound.stat().st_size)
            possiblefiles.append(filesfound)

        if any(possiblefiles):

            # get EEG data
            raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count, day_val, settings)

            # % Cue Start trig.cuestart(attended
            # trig.cuestart(i.spaceattd,i.featattd)
            #  1 - black | 2 - white | 1 - \ | 2 - / '
            # trig.cuestart = [
            #     111 112
            #     113 114];

            # Epoch to events of interest
            event_id = {'Black/Left_diag': 111, 'White/Left_diag': 112,
                        'Black/Right_diag': 113, 'White/Right_diag': 114}  # will be different triggers for training days

            epochs = mne.Epochs(eeg_data_interp, events, event_id=event_id, tmin=settings.timelimits_zeropad[0],
                                tmax=settings.timelimits_zeropad[1],
                                baseline=(0, 1 / settings.samplingfreq), picks=np.arange(settings.num_electrodes),
                                reject=dict(eeg=400), detrend=1)  #

            # drop bad channels
            epochs.drop_bad()
            epochs.plot_drop_log()
            # epochs2 = epochs.equalize_event_counts(event_id, method='mintime')

            # visualise topo
            # epochs.plot_psd_topomap()

            # get data for each  condition
            epochs_days[0:sum(elem == [] for elem in epochs['Black/Left_diag'].drop_log), :, :, 0, 0, day_count] = epochs[
                'Black/Left_diag'].get_data()
            epochs_days[0:sum(elem == [] for elem in epochs['Black/Right_diag'].drop_log), :, :, 0, 1, day_count] = epochs[
                'Black/Right_diag'].get_data()
            epochs_days[0:sum(elem == [] for elem in epochs['White/Left_diag'].drop_log), :, :, 1, 0, day_count] = epochs[
                'White/Left_diag'].get_data()
            epochs_days[0:sum(elem == [] for elem in epochs['White/Right_diag'].drop_log), :, :, 1, 1, day_count] = epochs[
                'White/Right_diag'].get_data()

        else:
            continue

    # average across trials
    erps_days = np.squeeze(np.nanmean(epochs_days, axis=0))
    erps_days_wave = np.squeeze(np.nanmean(epochs_days, axis=0))
    timepoints_zp = epochs.times

    fftdat, fftdat_epochs, freq = getSSVEPs(erps_days, epochs_days, epochs, settings, bids)
    fftdat_epochs = np.nanmean(fftdat_epochs, axis=0)  # average across trials to get the same shape for single trial SSVEPs

    # get signal to noise
    fftdat_snr = getfft_sigtonoise(settings, epochs, fftdat, freq)
    fftdat_snr_epochs = getfft_sigtonoise(settings, epochs, fftdat_epochs, freq)

    # get ssvep amplitudes
    SSVEPs_prepost, SSVEPs_prepost_channelmean, BEST = getSSVEPS_conditions(settings, fftdat, freq)  # trial average
    SSVEPs_prepost_epochs, SSVEPs_prepost_channelmean_epochs, BEST_epochs = getSSVEPS_conditions(settings, fftdat_epochs, freq)  # single trial

    # get ssvep amplitudes SNR
    SSVEPs_prepost_snr, SSVEPs_prepost_channelmean_snr, BEST_snr = getSSVEPS_conditions(settings, fftdat_snr, freq)  # trial average
    SSVEPs_prepost_epochs_snr, SSVEPs_prepost_channelmean_epochs_snr, BEST_epochs_snr = getSSVEPS_conditions(settings, fftdat_snr_epochs, freq)  # single trial

    # Plot
    ERPstring = 'ERP'
    plotResultsPrePost_subjects(SSVEPs_prepost_channelmean, settings, ERPstring, bids)

    ERPstring = 'Single Trial'
    plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs, settings, ERPstring, bids)

    # ERPstring = 'ERP SNR'
    # plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_snr, settings, ERPstring, bids)
    #
    # ERPstring = 'Single Trial SNR'
    # plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs_snr, settings, ERPstring, bids)


    # Get SSVEPs to Neurofeedback
    SSVEPsNF, SSVEPsNF_topo = getSSVEPS_neurofeedbackflick(settings, fftdat, freq)
    SSVEPsNF_epochs, SSVEPsNF_epochs_topo = getSSVEPS_neurofeedbackflick(settings, fftdat_epochs, freq)

    ERPstring = 'ERP'
    plot_NFSSVEP_subjects(SSVEPsNF, settings, ERPstring, bids)

    ERPstring = 'Single Trial'
    plot_NFSSVEP_subjects(SSVEPsNF_epochs, settings, ERPstring, bids)

    # Save
    np.savez(bids.direct_results / Path(bids.substring + "EEG_duringNF_results"),
             SSVEPsNF=SSVEPsNF,
             SSVEPsNF_epochs=SSVEPsNF_epochs,
             SSVEPs_prepost_channelmean_epochs=SSVEPs_prepost_channelmean_epochs,
             SSVEPs_prepost_channelmean=SSVEPs_prepost_channelmean,
             SSVEPsNF_topo=SSVEPsNF_topo,
             SSVEPsNF_epochs_topo=SSVEPsNF_epochs_topo,
             SSVEPs_prepost=SSVEPs_prepost,
             SSVEPs_prepost_epochs=SSVEPs_prepost_epochs,
             timepoints_zp=timepoints_zp,
             erps_days_wave=erps_days_wave,
             fftdat=fftdat, fftdat_epochs=fftdat_epochs, freq=freq)


def getSSVEPs(erps_days, epochs_days, epochs, settings, bids):
    from scipy.fft import fft, fftfreq, fftshift
    import matplotlib.pyplot as plt
    # for fft - zeropad
    zerotimes = np.where(
        np.logical_or(epochs.times < settings.timelimits[0], epochs.times > settings.timelimits[1]))
    erps_days[:, zerotimes, :, :, :] = 0
    epochs_days[:, :, zerotimes, :, :, :] = 0

    # for fft - fft
    fftdat = np.abs(fft(erps_days, axis=1)) / len(epochs.times)
    fftdat_epochs = np.abs(fft(epochs_days, axis=2)) / len(epochs.times)

    ## plot ERP FFT spectrum

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.mean(fftdat, axis=0)

    for day_count in np.arange(settings.num_days):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        if (day_count == 2): axuse = ax3

        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Black/Left_diag', color=settings.lightteal)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Black/Right_diag', color=settings.medteal)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='White/Left_diag', color=settings.darkteal)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='White/Right_diag', color=settings.yellow)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, .5)
        axuse.set_title(settings.string_testday[day_count])
        axuse.legend()

    titlestring = bids.substring + 'ERP FFT Spectrum during Neurofeedback'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # plot single trial FFT spectrum
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.nanmean(np.mean(fftdat_epochs, axis=1), axis=0)

    for day_count in np.arange(settings.num_days):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        if (day_count == 2): axuse = ax3
        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Black/Left_diag', color=settings.lightteal)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Black/Right_diag', color=settings.darkteal)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='White/Left_diag', color=settings.orange)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='White/Right_diag', color=settings.yellow)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, 0.5)
        axuse.set_title(settings.string_testday[day_count])
        axuse.legend()

    titlestring = bids.substring + 'Single Trial FFT Spectrum during Neurofeedback'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    return fftdat, fftdat_epochs, freq


def getfft_sigtonoise(settings, epochs, fftdat, freq):

    # get SNR fft
    numsnr = 10
    extradatapoints = (numsnr*2 + 1)
    snrtmp = np.empty((settings.num_electrodes, len(epochs.times)+extradatapoints , settings.num_attnstates, settings.num_levels, settings.num_days, extradatapoints ))
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


def plotResultsPrePost_subjects(SSVEPs_prepost_mean, settings, ERPstring, bids):
    import matplotlib.pyplot as plt
    from pathlib import Path

    # plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    labels = settings.string_attd_unattd
    x = np.arange(len(labels))
    width = 0.25

    for attn in np.arange(settings.num_attnstates):
        if (attn == 0): axuse = ax1
        if (attn == 1): axuse = ax2

        axuse.bar(x - width, SSVEPs_prepost_mean[:, 0, attn], width, label=settings.string_testday[0],
                  facecolor=settings.lightteal)  # Day 1
        axuse.bar(x , SSVEPs_prepost_mean[:, 1, attn], width, label=settings.string_testday[1],
                  facecolor=settings.medteal)  # Day 2
        axuse.bar(x + width , SSVEPs_prepost_mean[:, 2, attn], width, label=settings.string_testday[2],
                  facecolor=settings.darkteal)  # Day 3

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axuse.set_ylabel('SSVEP amp (µV)')
        axuse.set_title(settings.string_cuetype[attn])
        axuse.set_xticks(x)
        axuse.set_xticklabels(labels)
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = bids.substring + ' ' + ERPstring + ' SSVEPs during NF'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # next step - compute differences and plot
    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_cuetype
    x = np.arange(len(labels))
    width = 0.25

    day = 0
    datplot = SSVEPs_prepost_mean[0, day, :] - SSVEPs_prepost_mean[1, day, :]
    plt.bar(x - width , datplot, width, label=settings.string_testday[day], facecolor=settings.yellow)  # 'space'

    day = 1
    datplot = SSVEPs_prepost_mean[0, day, :] - SSVEPs_prepost_mean[1, day, :]
    plt.bar(x, datplot, width, label=settings.string_testday[day], facecolor=settings.orange)  # 'feature'

    day = 2
    datplot = SSVEPs_prepost_mean[0, day, :] - SSVEPs_prepost_mean[1, day, :]
    plt.bar(x + width , datplot, width, label=settings.string_testday[day], facecolor=settings.lightteal)  # 'feature'

    plt.ylabel('Delta SSVEP amp (µV)')
    plt.title(settings.string_cuetype[attn])
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = bids.substring + ' ' + ERPstring + ' SSVEP selectivity during NF'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')


def getSSVEPS_neurofeedbackflick(settings, fftdat, freq):
    # get indices for frequencies of interest

    hz_attn_index = np.argmin(np.abs(freq - settings.hz_NF))

    # Get best electrodes for each frequency
    BEST = np.empty((settings.num_best, settings.num_days))
    for day_count, day_val in enumerate(settings.daysuse):
        tmp = np.mean(np.mean(fftdat[:, hz_attn_index.astype(int), :, :, day_count], axis=1), axis=1) # average across cue conditions to get best electrodes for frequencies
        BEST[:, day_count] = tmp.argsort()[-settings.num_best:]
        # print(BEST[:, day_count])

    # Get SSVEPs for each frequency
    SSVEPsNF = np.empty((settings.num_attnstates, settings.num_levels, settings.num_days))
    SSVEPsNF_topo = np.empty((settings.num_electrodes,  settings.num_attnstates, settings.num_levels, settings.num_days))
    for day_count, day_val in enumerate(settings.daysuse):
        for cuetype in np.arange(2):
            for level in np.arange(2):
                bestuse = BEST[:,day_count].astype(int)
                hzuse = hz_attn_index.astype(int)

                SSVEPsNF[cuetype, level, day_count] = np.mean(fftdat[bestuse, hzuse, cuetype, level, day_count], axis=0)
                SSVEPsNF_topo[:, cuetype, level,day_count] = fftdat[:, hzuse, cuetype, level, day_count]

    return SSVEPsNF, SSVEPsNF_topo


def plot_NFSSVEP_subjects(SSVEPsNF, settings, ERPstring, bids):
    import matplotlib.pyplot as plt
    from pathlib import Path
    # import numpy as np

    # next step - compute differences and plot
    fig, ax = plt.subplots(figsize=(5, 5))

    # settings
    labels = settings.string_cuetype
    x = np.arange(len(labels))
    width = 0.25
    modifier = [-1, 0, 1]
    color = [settings.yellow, settings.lightteal, settings.darkteal]
    for day in np.arange(3):
        datplot = SSVEPsNF[:, :, day].mean(axis=1) # np.empty((settings.num_attnstates, settings.num_levels, settings.num_days))
        plt.bar(x + width*modifier[day], datplot, width, label=settings.string_testday[day], facecolor=color[day])  # 'space'

    plt.ylabel('SSVEP amp (µV)')
    plt.xlabel('CueType')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = bids.substring + ' ' + ERPstring + ' SSVEP to feedback'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')


def plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring, settings, freq):
    import matplotlib.pyplot as plt
    from pathlib import Path

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 10))

    chanmeanfft = np.mean(fftdat_ave, axis=0)

    for day_count in np.arange(settings.num_days):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        if (day_count == 2): axuse = ax3

        hz_attn = settings.hz_attn  # ['\B \W'], ['/B /W']
        axuse.axvline(hz_attn[0, 0], 0, 1, linestyle='--', color='k', alpha=0.2)  # black/Left_diag
        axuse.annotate("Black-leftdiag", (hz_attn[0, 0], 0.3))
        axuse.axvline(hz_attn[0, 1], 0, 1, linestyle='--', color='k', alpha=0.2)  # black/Right_diag
        axuse.annotate("Black-rightdiag", (hz_attn[1, 0], 0.3))
        axuse.axvline(hz_attn[1, 0], 0, 1, linestyle='--', color='k', alpha=0.2)  # white/Left_diag
        axuse.annotate("white-leftdiag", (hz_attn[0, 1], 0.3))
        axuse.axvline(hz_attn[1, 1], 0, 1, linestyle='--', color='k', alpha=0.2)  # white/Right_diag
        axuse.annotate("white-rightdiag", (hz_attn[1, 1], 0.3))

        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Black/Left_diag', color=settings.lightteal)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Black/Right_diag', color=settings.darkteal)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='White/Left_diag', color=settings.yellow)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='White/Right_diag', color=settings.orange)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, .4)
        axuse.set_title(settings.string_testday[day_count])
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = 'During NF Group Mean '+ ERPstring +' FFT Spectrum' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


def plotGroupSSVEPs(SSVEPs_group, bids, ERPstring, settings):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import helperfunctions_ATTNNF as helper

    M = np.nanmean(SSVEPs_group, axis=3)

    E = np.empty((settings.num_attnstates, settings.num_days, settings.num_attnstates))
    for attn in np.arange(settings.num_attnstates):
        for day in np.arange(settings.num_days):
            E[:,day, attn] = helper.within_subjects_error(SSVEPs_group[:,day,attn,:].T)

    # E = np.std(SSVEPs_group, axis=3) / settings.num_subs

    # plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    labels = settings.string_attd_unattd
    x = np.arange(len(labels))
    width = 0.25

    for attn in np.arange(settings.num_attnstates):
        if (attn == 0): axuse = ax1
        if (attn == 1): axuse = ax2

        axuse.bar(x - width, M[:, 0, attn], width, label=settings.string_testday[0],
                  facecolor=settings.lightteal)  # Day 1
        axuse.bar(x , M[:, 1, attn], width, label=settings.string_testday[1],
                  facecolor=settings.medteal)  # Day 2
        axuse.bar(x + width , M[:, 2, attn], width, label=settings.string_testday[2],
                  facecolor=settings.darkteal)  # Day 3

        # Add some text for labels, title and custom x-axis tick labels, etc.
        axuse.set_ylabel('SSVEP amp (µV)')
        axuse.set_title(settings.string_cuetype[attn])
        axuse.set_xticks(x)
        axuse.set_xticklabels(labels)
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = 'Group Mean ' + ERPstring + ' SSVEPs during NF TRAIN ' + settings.string_attntrained[
        settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # compute SSVEP differences and plot
    diffdat = SSVEPs_group[0, :, :, :] - SSVEPs_group[1, :, :, :]
    M = np.nanmean(diffdat, axis=2)
    # E = np.std(diffdat, axis=2) / settings.num_subs

    E = np.empty(( settings.num_days, settings.num_attnstates))
    for day in np.arange(settings.num_days):
        E[day, :] = helper.within_subjects_error(diffdat[day,:,:].T)


    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_cuetype
    x = np.arange(len(labels))
    width = 0.25
    xpos = np.array((x - width , x, x + width ))
    colors = np.array((settings.yellow, settings.orange, settings.lightteal))
    for day in np.arange(settings.num_days):
        ax.bar(xpos[day], M[day, :], yerr=E[day, :], width=width, label=settings.string_testday[day],
               facecolor=colors[day])


    ax.plot(xpos[:, 0], diffdat[:, 0, :], '-', alpha=0.3, color='k')
    ax.plot(xpos[:, 1], diffdat[:, 1, :], '-', alpha=0.3, color='k')

    plt.ylabel('Delta SSVEP amp (µV)')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = 'Group Mean ' + ERPstring + ' SSVEP selectivity during NF TRAIN ' + settings.string_attntrained[
        settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


def plotGroupSSVEPs_NFtagging(SSVEPsNF_group, bids, ERPstring, settings):
    import matplotlib.pyplot as plt
    from pathlib import Path
    import helperfunctions_ATTNNF as helper

    datuse = SSVEPsNF_group.mean(axis=0).mean(axis=0) #  np.empty((settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    M = np.nanmean( datuse, axis=1) #  np.empty((settings.num_attnstates, settings.num_days, num_subs))
    E = helper.within_subjects_error(datuse.T)

    fig, ax = plt.subplots(figsize=(5, 5))
    width = 0.25
    xuse = [-width, 0, width]
    colors = np.array((settings.lightteal, settings.medteal, settings.darkteal))
    for day in np.arange(settings.num_days):
        ax.bar(xuse[day], M[day], yerr=E[day], width=width, label=settings.string_testday[day], facecolor=colors[day])

    ax.plot(xuse, datuse, '-', alpha=0.3, color='k')

    plt.ylabel('SSVEP amp (µV)')
    plt.xticks(xuse, settings.string_testday[:-1])
    plt.legend()
    ax.set_frame_on(False)

    titlestring = 'Group Mean ' + ERPstring + ' SSVEP to feedback during NF TRAIN ' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


def collateEEG_duringNF(settings):
    print('collating SSVEP amplitudes during NF')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_duringNF()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(
        settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))

    SSVEPsNF_group = np.empty((settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    SSVEPsNF_epochs_group = np.empty((settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))

    fftdat_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates,settings.num_levels, settings.num_days, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates,settings.num_levels, settings.num_days, num_subs))

    SSVEPs_group[:] = np.nan
    SSVEPs_epochs_group[:] = np.nan
    SSVEPsNF_group[:] = np.nan
    SSVEPsNF_epochs_group[:] = np.nan
    fftdat_group[:] = np.nan
    fftdat_epochs_group[:] = np.nan

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_duringNF_results.npz"), allow_pickle=True)
        # saved vars: SSVEPsNF, SSVEPsNF_epochs, SSVEPs_prepost_channelmean_epochs, SSVEPs_prepost_channelmean, SSVEPsNF_topo, SSVEPsNF_epochs_topo, SSVEPs_prepost,
        #SSVEPs_prepost_epochs, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq

        # store results
        SSVEPs_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean']
        SSVEPs_epochs_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']

        SSVEPsNF_group[:, :, :, sub_count] = results['SSVEPsNF']
        SSVEPsNF_epochs_group[:, :, :, sub_count] = results['SSVEPsNF_epochs']

        fftdat_group[:, :, :, :, :, sub_count] = results['fftdat']
        fftdat_epochs_group[:, :, :, :, :, sub_count] = results['fftdat_epochs']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

    # plot grand average frequency spectrum
    fftdat_ave = np.nanmean(fftdat_group, axis=5)
    plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring='ERP', settings=settings, freq=freq)

    fftdat_epochs_ave = np.nanmean(fftdat_epochs_group, axis=5)
    plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    plotGroupSSVEPs(SSVEPs_group, bids, ERPstring='ERP', settings=settings)
    plotGroupSSVEPs(SSVEPs_epochs_group, bids, ERPstring='Single Trial', settings=settings)

    # plot average SSVEP results for neurofeedback tagging
    plotGroupSSVEPs_NFtagging(SSVEPsNF_group, bids, ERPstring='ERP', settings=settings)
    plotGroupSSVEPs_NFtagging(SSVEPsNF_epochs_group, bids, ERPstring='Single Trial', settings=settings)


def plotgroupedresult_complex_duringNF(df_grouped, measurestring, bids, coloruse, ylims):

    fig, (ax1) = plt.subplots(1, 1, figsize=(10, 6))

    # Reaction time Grouped violinplot
    colors = coloruse

    sns.swarmplot(x="TrainingGroup", y=measurestring, hue="Testday", dodge=True, data=df_grouped, color="0", alpha=0.3)
    sns.violinplot(x="TrainingGroup", y=measurestring, hue="Testday", data=df_grouped, palette=sns.color_palette(colors), style="ticks", ax=ax1,
                   inner="box", alpha=0.6)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:3], labels[:3])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(ylims)

    titlestring = 'Motion Task SSVEPs ' + measurestring + ' by Day during NF'
    plt.suptitle(titlestring)

    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')


def collateEEG_duringNF_compare(settings):
    import pandas as pd
    print('collating SSVEP amplitudes pre Vs. post training compareing Space Vs. Feat Training')

    # preallocate
    subID = list()
    subIDval = list()
    daystrings = list()
    TrainingGroup = list()

    SSVEPs_feat_attd = list()
    SSVEPs_feat_unattd = list()
    SSVEP_feat_Selectivity = list()
    SSVEPs_space_attd = list()
    SSVEPs_space_unattd = list()
    SSVEP_space_Selectivity = list()
    SSVEP_NF = list()

    # cycle trough space and feature train groups
    for attntrained, attntrainedstr in enumerate(settings.string_attntrained):  # cycle trough space, feature and sham train groups
        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_EEG_prepost()

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "EEG_duringNF_results.npz"), allow_pickle=True)

            # store results
            # SSVEPs_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean'] # np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
            # SSVEPsNF_group[:, :, :, sub_count] = results['SSVEPsNF'] #  np.empty((settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
            for testday, daystring in enumerate(settings.string_testday[:-1]):
                subID.append(bids.substring)
                subIDval.append(sub_count + 37*attntrained)
                daystrings.append(daystring)
                TrainingGroup.append(attntrainedstr)

                SSVEPS_space = results['SSVEPs_prepost_channelmean'][:, testday, 0]
                SSVEPs_space_attd.append(SSVEPS_space[0])
                SSVEPs_space_unattd.append(SSVEPS_space[1])
                SSVEP_space_Selectivity.append(SSVEPS_space[0] - SSVEPS_space[1])

                SSVEPS_feat = results['SSVEPs_prepost_channelmean'][:, testday, 1] #_epochs
                SSVEPs_feat_attd.append(SSVEPS_feat[0])
                SSVEPs_feat_unattd.append(SSVEPS_feat[1])
                SSVEP_feat_Selectivity.append(SSVEPS_feat[0] - SSVEPS_feat[1])

                dat = results['SSVEPsNF_epochs'][:, :, testday].mean(axis=0).mean(axis=0)
                SSVEP_NF.append(dat)

    # Dataframe
    data = {'SubID': subID, 'SubIDval': subIDval, 'Testday': daystrings, 'TrainingGroup': TrainingGroup,
            'SSVEPs_space_attd': SSVEPs_space_attd, 'SSVEPs_space_unattd':  SSVEPs_space_unattd, 'SSVEP_space_Selectivity':  SSVEP_space_Selectivity,
            'SSVEPs_feat_attd': SSVEPs_feat_attd, 'SSVEPs_feat_unattd':  SSVEPs_feat_unattd, 'SSVEP_feat_Selectivity':  SSVEP_feat_Selectivity,'SSVEP_NF': SSVEP_NF}
    df_selectivity = pd.DataFrame(data)

    # # lets run some stats with R - save it out
    df_selectivity.to_csv(bids.direct_results_group_compare / Path("motiondiscrim_SelectivityResults_NF_ALL.csv"), index=False)

    # Get Mean amplitudes
    df_selectivity["SSVEPs_space_mean"] = (df_selectivity["SSVEPs_space_attd"] + df_selectivity["SSVEPs_space_unattd"]) / 2
    df_selectivity["SSVEPs_feat_mean"] = (df_selectivity["SSVEPs_feat_attd"] + df_selectivity["SSVEPs_feat_unattd"]) / 2

    # Plot results
    coloruse = [settings.lightteal, settings.medteal, settings.darkteal]
    plotgroupedresult_complex_duringNF(df_selectivity, "SSVEPs_space_mean", bids, coloruse, [0, 0.5])
    plotgroupedresult_complex_duringNF(df_selectivity, "SSVEPs_feat_mean", bids, coloruse, [0, 0.5])

    plotgroupedresult_complex_duringNF(df_selectivity, "SSVEP_space_Selectivity", bids, coloruse, [-0.15, 0.35])
    plotgroupedresult_complex_duringNF(df_selectivity, "SSVEP_feat_Selectivity", bids, coloruse, [-0.1, 0.15])

    plotgroupedresult_complex_duringNF(df_selectivity, "SSVEP_NF", bids, coloruse, [-0.2, 1.1])