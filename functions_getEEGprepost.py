import numpy as np
from pathlib import Path
import mne

def get_eeg_data(bids, day_count):
    import mne
    import matplotlib.pyplot as plt
    # decide which EEG file to use
    possiblefiles = []
    filesizes = []
    for filesfound in bids.direct_data_eeg.glob(bids.filename_eeg + "*.eeg"):
        filesizes.append(filesfound.stat().st_size)

    for filesfound in bids.direct_data_eeg.glob(bids.filename_eeg + "*.vhdr"):
        possiblefiles.append(filesfound)

    file2useIDX = np.argmax(filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
    file2use = possiblefiles[file2useIDX]

    montage = {'Iz': [-28.6, -109.8, -28.0],
               'Oz': [-28.2, -107.8, 8.5],
               'POz': [-18.2, -97.5, 44.6],
               'O1': [52.7, -94.0, -34.1],
               'O2': [0.1, -110.1, 14.0],
               'PO3': [-46.4, -95.2, 20.7],
               'PO4': [19.1, -97.6, 44.5],
               'PO7': [-52.6, -94.0, -34.0],
               'PO8': [47.8, -95.3, 20.8] }

    montageuse = mne.channels.make_dig_montage(ch_pos=montage, lpa=[-82.5, -19.2, -46], nasion=[0, 83.2, -38.3], rpa=[82.2, -19.2, -46])

    # load EEG file
    raw = mne.io.read_raw_brainvision(file2use, preload=True, scale=1e6)
    print(raw.info)

    # raw.plot(remove_dc = False, scalings=dict(eeg=50))

    # pick events
    events = mne.find_events(raw, stim_channel="TRIG")
    mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp)
    print('Found %s events, first five:' % len(events))
    print(events[:5])

    # set bad chans
    if (np.logical_and(bids.substring == 'sub-02', day_count == 0)):
        raw.info['bads'] = ['O1']
    if (np.logical_and(bids.substring == 'sub-09', day_count == 0)):
        raw.info['bads'] = ['PO4']
    if (np.logical_and(bids.substring == 'sub-23', day_count == 0)):
        raw.info['bads'] = ['Iz']
    if (np.logical_and(bids.substring == 'sub-10', day_count == 1)):
        raw.info['bads'] = ['Oz']

    plt.show()
    # tmp = input('check the eeg data')

    eeg_data = raw.copy().pick_types(eeg=True, exclude=['TRIG'])
    eeg_data.info.set_montage(montageuse)
    eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=False)

    # Filter Data
    eeg_data_interp.filter(l_freq=1, h_freq=45, h_trans_bandwidth=0.1)

    #plot results
    eeg_data_interp.plot(remove_dc=False, scalings=dict(eeg=50))

    return raw, events, eeg_data_interp

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
    # to do: make subplots for days, set titles and x and y lablels, set ylim to be comparable, get rid of zero line, save

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
        axuse.set_ylim(0, .5)
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
    SSVEPs_prepost[:, :, :, 0] = np.mean(spaceSSVEPs, axis=3)
    SSVEPs_prepost[:, :, :, 1] = np.mean(featureSSVEPs, axis=3)

    # get best electrodes to use and mean SSVEPs for these electrodes
    BEST = np.empty((settings.num_best, settings.num_days, settings.num_attnstates))
    SSVEPs_prepost_mean = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates))
    for day_count, day_val in enumerate(settings.daysuse):
        for attn_count, attn_val in enumerate(settings.string_attntrained):
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

    labels = settings.string_attntrained
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
    M = np.mean(SSVEPs_prepost_group, axis=3)
    E = np.std(SSVEPs_prepost_group, axis=3) / settings.num_subs

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
    E = np.std(diffdat, axis=2) / settings.num_subs

    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_attntrained
    x = np.arange(len(labels))
    width = 0.35
    xpos = np.array((x - width / 2, x + width / 2))
    colors = np.array((settings.yellow, settings.orange))
    for day in np.arange(settings.num_days):
        ax.bar(xpos[day], M[day, :], yerr=E[day, :], width=width, label=settings.string_prepost[day],
               facecolor=colors[day])
        ax.plot(xpos[day] + np.random.random((settings.num_subs, settings.num_attnstates)) * width * 0.5 - width * 0.5 / 2,
                diffdat[day, :, :].T, '.', alpha=0.3, color='k')

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