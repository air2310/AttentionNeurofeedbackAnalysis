import numpy as np
from pathlib import Path
import mne


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


    # get ssveps for attended and unattended space condition, averaging across black and white cues - in this condition, we use combined feature space cues, so both a space and a feature are cued at once.
    left_diag, right_diag =  0, 1

    spaceSSVEPs = np.empty((settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_spaces))
    for level_count, level in enumerate(['Left_diag', 'Right_diag']): # cycle through space trials on which the left and right diag were cued

        left_tmp = np.mean(fftdat[:, hz_attn_index[left_diag, :].astype(int), :, :, :],axis=1)
        right_tmp = np.mean(fftdat[:, hz_attn_index[right_diag, :].astype(int), :, :, :], axis=1)

        if (level == 'Left_diag'): # when left diag cued
            attendedSSVEPs =   np.mean( left_tmp[:,:,level_count,:], axis=1)  # average across left_diag frequencies at both features (black, white)
            unattendedSSVEPs = np.mean( right_tmp[:,:,level_count,:], axis=1)  # average across right_diag frequencies at both features (black, white) when both features are cued
            print(attendedSSVEPs.shape)
            print(left_tmp.shape)
        if (level == 'Right_diag'): # whien right diag cued
            attendedSSVEPs = np.mean( right_tmp[:,:,level_count,:], axis=1) # average across right_diag frequencies at both features (black, white)
            unattendedSSVEPs =  np.mean( left_tmp[:,:,level_count,:], axis=1)  # average across left_diag frequencies at both features (black, white) when both features are cued
        #
            print(attendedSSVEPs.shape)
            print(left_tmp.shape)
        spaceSSVEPs[:, :, 0, level_count] = attendedSSVEPs
        spaceSSVEPs[:, :, 1, level_count] = unattendedSSVEPs

    # get ssveps for attended and unattended feature condition, averaging across left and right diag cues - in this condition, we use combined feature space cues, so both a space and a feature are cued at once.

    black, white = 0, 1
    featureSSVEPs = np.empty(
        (settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_features))
    for level_count, level in enumerate(['Black', 'White']): # average through trials on which black and white were cued
        if (level == 'Black'):
            attendedSSVEPs = np.mean(np.mean(fftdat[:, hz_attn_index[:, black].astype(int), level_count, :, :],
                                             axis=1), axis=1)  # average across black frequencies at both spaces (\, /) when both spaces are cued
            unattendedSSVEPs = np.mean(np.mean(fftdat[:, hz_attn_index[:, white].astype(int), level_count, :, :],
                                               axis=1), axis=1)  # average across white frequencies at both spaces (\, /) when both spaces are cued

        if (level == 'White'):
            attendedSSVEPs = np.mean(np.mean(fftdat[:, hz_attn_index[:, white].astype(int), level_count, :, :],
                                             axis=1), axis=1)  # average across black frequencies at both spaces (\, /) when both spaces are cued
            unattendedSSVEPs = np.mean(np.mean(fftdat[:, hz_attn_index[:, black].astype(int), level_count, :, :],
                                               axis=1), axis=1)  # average across white frequencies at both spaces (\, /) when both spaces are cued

        featureSSVEPs[:, :, 0, level_count] = attendedSSVEPs
        featureSSVEPs[:, :, 1, level_count] = unattendedSSVEPs

    # average across cue types and store the SSVEPs alltogether for plotting and further analysis
    SSVEPs_prepost = np.empty((settings.num_electrodes, settings.num_days, settings.num_attd_unattd, settings.num_attnstates))
    SSVEPs_prepost[:, :, :, 0] = np.mean(spaceSSVEPs, axis=3)
    SSVEPs_prepost[:, :, :, 1] = np.mean(featureSSVEPs, axis=3)

    # get best electrodes to use and mean SSVEPs for these electrodes
    BEST = np.empty((settings.num_best, settings.num_days, settings.num_attnstates))
    SSVEPs_prepost_mean = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates))
    for day_count, day_val in enumerate(settings.daysuse):
        for attn_count, attn_val in enumerate(settings.string_attntrained):
            tmp = np.mean(SSVEPs_prepost[:, day_count, :, attn_count], axis=1)
            BEST[:, day_count, attn_count] = tmp.argsort()[-settings.num_best:]

            SSVEPs_prepost_mean[:, day_count, attn_count] = np.mean(SSVEPs_prepost[BEST[:, day_count, attn_count].astype(int), day_count, :, attn_count], axis=0)


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

    labels = settings.string_attntrained
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
    import Analysis_Code.helperfunctions_ATTNNF as helper

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

    labels = settings.string_attntrained
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
