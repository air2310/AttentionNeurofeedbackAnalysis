import numpy as np
from pathlib import Path
import mne
import helperfunctions_ATTNNF as helper

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
        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # get EEG data
        raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count, settings)

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

       # average across trials
    erps_days = np.squeeze(np.nanmean(epochs_days, axis=0))
    erps_days_wave = np.squeeze(np.nanmean(epochs_days, axis=0))
    timepoints_zp = epochs.times

    # # Get SSVEPs
    fftdat, fftdat_epochs, freq = getSSVEPs(erps_days, epochs_days, epochs, settings, bids)
    fftdat_epochs = np.nanmean(fftdat_epochs,
                               axis=0)  # average across trials to get the same shape for single trial SSVEPs

    # get signal to noise
    fftdat_snr = getfft_sigtonoise(settings, epochs, fftdat, freq)
    fftdat_snr_epochs = getfft_sigtonoise(settings, epochs, fftdat_epochs, freq)

    # get ssvep amplitudes
    # @ David! This is what I'd like you to check :)
    SSVEPs_prepost, SSVEPs_prepost_channelmean, BEST = getSSVEPS_conditions(settings, fftdat,
                                                                                   freq)  # trial average
    SSVEPs_prepost_epochs, SSVEPs_prepost_channelmean_epochs, BEST_epochs = getSSVEPS_conditions(settings,
                                                                                                        fftdat_epochs,
                                                                                                        freq)  # single trial

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

    ERPstring = 'ERP SNR'
    plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_snr, settings, ERPstring, bids)

    ERPstring = 'Single Trial SNR'
    plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs_snr, settings, ERPstring, bids)



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
    # for fft - zeropad
    zerotimes = np.where(
        np.logical_or(epochs.times < settings.timelimits[0], epochs.times > settings.timelimits[1]))
    erps_days[:, zerotimes, :, :, :] = 0
    epochs_days[:, :, zerotimes, :, :, :] = 0

    # for fft - fft
    fftdat = np.abs(fft(erps_days, axis=1)) / len(epochs.times)
    fftdat_epochs = np.abs(fft(epochs_days, axis=2)) / len(epochs.times)

    ## plot ERP FFT spectrum

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
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
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.nanmean(np.mean(fftdat_epochs, axis=1), axis=0)

    for day_count in np.arange(2):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Space/Left_diag', color=settings.lightteal)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Space/Right_diag', color=settings.medteal)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='Feat/Black', color=settings.darkteal)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='Feat/White', color=settings.yellow)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, 0.5)
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

    # get ssveps for space condition, sorted to represent attended vs. unattended
    cuetype = 0  # space
    left_diag, right_diag =  0, 1

    spaceSSVEPs = np.empty((settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_levels))
    for level_count, level in enumerate(['Left_diag', 'Right_diag']): # cycle through space trials on which the left and right diag were cued
        if (level == 'Left_diag'): # when left diag cued
            attendedSSVEPs = np.mean(fftdat[:, hz_attn_index[left_diag, :].astype(int), cuetype, level_count, :],
                                     axis=1)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[right_diag, :].astype(int), cuetype, level_count, :],
                                       axis=1)  # average across right_diag frequencies at both features (black, white)

        if (level == 'Right_diag'): # whien right diag cued
            attendedSSVEPs = np.mean(fftdat[:, hz_attn_index[right_diag, :].astype(int), cuetype, level_count, :],
                                     axis=1)  # average across right_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[left_diag, :].astype(int), cuetype, level_count, :],
                                       axis=1)  # average across left_diag frequencies at both features (black, white)

        spaceSSVEPs[:, :, 0, level_count] = attendedSSVEPs
        spaceSSVEPs[:, :, 1, level_count] = unattendedSSVEPs

    # get ssveps for feature condition, sorted to represent attended vs. unattended
    cuetype = 1  # feature
    black, white = 0, 1
    featureSSVEPs = np.empty(
        (settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_levels))
    for level_count, level in enumerate(['Black', 'White']): # average through trials on which black and white were cued
        if (level == 'Black'):
            attendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, black].astype(int), cuetype, level_count, :],
                                     axis=1)  # average across black frequencies at both spatial positions
            unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, white].astype(int), cuetype, level_count, :],
                                       axis=1)  # average across white frequencies at both spatial positions

        if (level == 'White'):
            attendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, white].astype(int), cuetype, level_count, :],
                                     axis=1)  # average across white frequencies at both spatial positions
            unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, black].astype(int), cuetype, level_count, :],
                                       axis=1)  # average across black frequencies at both spatial positions

        featureSSVEPs[:, :, 0, level_count] = attendedSSVEPs
        featureSSVEPs[:, :, 1, level_count] = unattendedSSVEPs

    # average across cue types and store the SSVEPs alltogether for plotting and further analysis
    SSVEPs_prepost = np.empty(
        (settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_attnstates))
    SSVEPs_prepost[:, :, :, 0] = np.mean(spaceSSVEPs, axis=3) # chans, # daycount, # attd unattd, # space/feat
    SSVEPs_prepost[:, :, :, 1] = np.mean(featureSSVEPs, axis=3)

    # get best electrodes to use and mean SSVEPs for these electrodes
    BEST = np.empty((settings.num_best, settings.num_days, settings.num_attnstates))
    SSVEPs_prepost_mean = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates))
    for day_count, day_val in enumerate(settings.daysuse):
        for attn_count, attn_val in enumerate(settings.string_cuetype):
            tmp = np.mean(SSVEPs_prepost[:, day_count, :, attn_count], axis=1)
            BEST[:, day_count, attn_count] = tmp.argsort()[-settings.num_best:]

            SSVEPs_prepost_mean[:, day_count, attn_count] = np.mean(
                SSVEPs_prepost[BEST[:, day_count, attn_count].astype(int), day_count, :, attn_count], axis=0)


    return SSVEPs_prepost, SSVEPs_prepost_mean, BEST

def get_wavelets_prepost(erps_days, settings, epochs, BEST, bids):
    import mne
    import matplotlib.pyplot as plt
    from pathlib import Path

    erps_days_wave = erps_days.transpose(4, 0, 1, 2, 3)  # [day, chans,time,cuetype, level]

    cuetype = 0  # space
    left_diag, right_diag = 0, 1

    spacewavelets = np.empty((len(epochs.times), settings.num_days, settings.num_attd_unattd, settings.num_levels))
    for level_count, level in enumerate(
            ['Left_diag', 'Right_diag']):  # cycle through space trials on which the left and right diag were cued
        if (level == 'Left_diag'):  # when left diag cued
            freqs2use_attd = settings.hz_attn[left_diag, :]
            freqs2use_unattd = settings.hz_attn[right_diag, :]

        if (level == 'Right_diag'):  # whien right diag cued
            freqs2use_attd = settings.hz_attn[right_diag, :]
            freqs2use_unattd = settings.hz_attn[left_diag, :]

        attended_wavelets = np.mean(mne.time_frequency.tfr_array_morlet(erps_days_wave[:, :, :, cuetype, level_count],settings.samplingfreq, freqs=freqs2use_attd,n_cycles=freqs2use_attd, output='power'),axis=2)  # average across left_diag frequencies at both features (black, white)
        unattended_wavelets = np.mean(mne.time_frequency.tfr_array_morlet(erps_days_wave[:, :, :, cuetype, level_count],settings.samplingfreq, freqs=freqs2use_unattd,n_cycles=freqs2use_unattd, output='power'),axis=2)  # average across right_diag frequencies at both features (black, white)

        for day_count in np.arange(settings.num_days):  # average across best electrodes for each day
            spacewavelets[:, day_count, 0, level_count] = np.mean(
                attended_wavelets[day_count, BEST[:, day_count, cuetype].astype(int), :], axis=0)  # attended freqs
            spacewavelets[:, day_count, 1, level_count] = np.mean(
                unattended_wavelets[day_count, BEST[:, day_count, cuetype].astype(int), :], axis=0)  # unattended freqs

    # feature wavelets
    cuetype = 1  # space
    black, white = 0, 1

    featurewavelets = np.empty((len(epochs.times), settings.num_days, settings.num_attd_unattd, settings.num_levels))

    for level_count, level in enumerate(
            ['Black', 'White']):  # average through trials on which black and white were cued
        if (level == 'Black'):
            freqs2use_attd= settings.hz_attn[:, black]
            freqs2use_unattd = settings.hz_attn[:, white]

        if (level == 'White'):
            freqs2use_attd = settings.hz_attn[:, white]
            freqs2use_unattd = settings.hz_attn[:, black]

        attended_wavelets = np.mean(mne.time_frequency.tfr_array_morlet(erps_days_wave[:, :, :, cuetype, level_count],settings.samplingfreq, freqs=freqs2use_attd,n_cycles=freqs2use_attd, output='power'),axis=2)  # average across left_diag frequencies at both features (black, white)
        unattended_wavelets = np.mean(mne.time_frequency.tfr_array_morlet(erps_days_wave[:, :, :, cuetype, level_count],settings.samplingfreq, freqs=freqs2use_unattd,n_cycles=freqs2use_unattd, output='power'),axis=2)  # average across right_diag frequencies at both features (black, white)

        for day_count in np.arange(settings.num_days):  # average across best electrodes
            featurewavelets[:, day_count, 0, level_count] = np.mean(
                attended_wavelets[day_count, BEST[:, day_count, cuetype].astype(int), :], axis=0)  # attended freqs
            featurewavelets[:, day_count, 1, level_count] = np.mean(
                unattended_wavelets[day_count, BEST[:, day_count, cuetype].astype(int), :], axis=0)  # unattended freqs

    # average across cue types and store the SSVEPs alltogether for plotting and further analysis
    wavelets_prepost = np.empty(
        (len(epochs.times), settings.num_days, settings.num_attd_unattd, settings.num_attnstates))
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
                dataplot = SSVEPs[:, day, attd, attntype]  # chans, # daycount, # attd unattd, # space/feat
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
                dataplot = SSVEPs_mean[:, day, attd, attntype]  # chans, # daycount, # attd unattd, # space/feat
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
        axuse.set_ylim(0, .4)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = 'Group Mean '+ ERPstring +' FFT Spectrum' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

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

