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

    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_motionepochs, settings.samplingfreq)

    # preallocate
    num_epochs = 240
    epochs_days = np.empty((int(num_epochs), settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_attd_unattd, settings.num_days))
    epochs_days[:] = np.nan

    # iterate through test days to get data
    for day_count, day_val in enumerate(settings.daysuse):
        print(day_val)
        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # get EEG data
        raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count, day_val, settings)

        # fix FASA/FUSA trigger system for pre-post trials (we implemented a trigger system based on the way triggers were presented during NF. described in detail in the program script).
        idx_feat = np.where(np.logical_or(events[:, 2] == 121, events[:, 2] == 122))
        idx_space = np.where(np.logical_or(events[:, 2] == 123, events[:, 2] == 124))

        tmp = np.zeros((len(events)))
        tmp[idx_space], tmp[idx_feat] = 1, 2
        cues = np.where(tmp>0)[0]

        for i in np.arange(len(cues)):
            # get events to edit
            if i == len(cues)-1:
                tmpevents = events[cues[i] + 1: -1, :]
            else:
                tmpevents = events[cues[i]+1 : cues[i+1]-1,:]

            # generate new triggers
            motiontrigs = np.where(np.isin(tmpevents[:,2], settings.trig_motiononset[:]))[0]

            for j in np.arange(len(motiontrigs)):
                cuetype = int(tmp[cues[i]]-1) # 0 = space, 1 = feat
                cuelevel = abs((events[cues[i], 2]-120) % 2 -1) # 0 = level 1, 1 = level 2

                # establish if this movement was in the cued or uncued dots
                FASA = tmpevents[j, 2] % 4 + 1
                if cuetype == 0:
                    if np.isin(FASA,[1,2]):
                        motioncued = 0
                    else:
                        motioncued = 1
                if cuetype == 1:
                    if np.isin(FASA, [1, 3]):
                        motioncued = 0
                    else:
                        motioncued = 1

                newtrig = settings.trig_motiononset_new[cuetype,cuelevel,motioncued]
                tmpevents[motiontrigs[j],2] = newtrig

            # replace edited triggers
            if i == len(cues) - 1:
                events[cues[i] + 1: -1, :] = tmpevents
            else:
                events[cues[i] + 1: cues[i + 1] - 1, :] = tmpevents

        # Epoch to events of interest
        event_id = {'Space/Left_diag/targetmotion': 1, 'Space/Left_diag/distractmotion': 5,
                    'Space/Right_diag/targetmotion': 3, 'Space/Right_diag/distractmotion': 7,
                    'Feat/Black/targetmotion': 2, 'Feat/Black/distractmotion': 6,
                    'Feat/White/targetmotion': 4, 'Feat/White/distractmotion': 8}  # will be different triggers for training days

        epochs = mne.Epochs(eeg_data_interp, events, event_id=event_id, tmin=settings.timelimits_motionepochs[0],
                            tmax=settings.timelimits_motionepochs[1],
                            baseline=(0, 1 / settings.samplingfreq), picks=np.arange(settings.num_electrodes),
                            reject=dict(eeg=400), detrend=1)  #

        # drop bad channels
        epochs.drop_bad()
        # epochs.plot_drop_log()
        # epochs2 = epochs.equalize_event_counts(event_id, method='mintime')

        # visualise topo
        # epochs.plot_psd_topomap()

        # get data for each  condition # settings.num_attnstates, settings.num_levels, settings.num_attd_unattd, settings.num_days
        epochs_days[0:sum(elem == [] for elem in epochs['Space/Left_diag/targetmotion'].drop_log), :, :, 0, 0, 0, day_count] = epochs['Space/Left_diag/targetmotion'].get_data()
        epochs_days[0:sum(elem == [] for elem in epochs['Space/Left_diag/distractmotion'].drop_log), :, :, 0, 0, 1, day_count] = epochs['Space/Left_diag/distractmotion'].get_data()

        epochs_days[0:sum(elem == [] for elem in epochs['Space/Right_diag/targetmotion'].drop_log), :, :, 0, 1, 0, day_count] = epochs['Space/Right_diag/targetmotion'].get_data()
        epochs_days[0:sum(elem == [] for elem in epochs['Space/Right_diag/distractmotion'].drop_log), :, :, 0, 1, 1, day_count] = epochs['Space/Right_diag/distractmotion'].get_data()

        epochs_days[0:sum(elem == [] for elem in epochs['Feat/Black/targetmotion'].drop_log), :, :, 1, 0, 0, day_count] = epochs['Feat/Black/targetmotion'].get_data()
        epochs_days[0:sum(elem == [] for elem in epochs['Feat/Black/distractmotion'].drop_log), :, :, 1, 0, 1, day_count] = epochs['Feat/Black/distractmotion'].get_data()

        epochs_days[0:sum(elem == [] for elem in epochs['Feat/White/targetmotion'].drop_log), :, :, 1, 1, 0, day_count] = epochs['Feat/White/targetmotion'].get_data()
        epochs_days[0:sum(elem == [] for elem in epochs['Feat/White/distractmotion'].drop_log), :, :, 1, 1, 1, day_count] = epochs['Feat/White/distractmotion'].get_data()

       # average across trials
    erps_days = np.squeeze(np.nanmean(epochs_days, axis=0))
    timepoints_zp = epochs.times

    fig = plt.subplots()
    datplot = np.mean(np.mean(np.mean(erps_days[:, :, 0, :, 1, :], axis=3), axis=2), axis=0)
    plt.plot(epochs.times, datplot)
    datplot = np.mean(np.mean(np.mean(erps_days[:, :, 0, :, 0, :], axis=3), axis=2), axis=0)
    plt.plot(epochs.times, datplot)

    # # Get SSVEPs
    fftdat, fftdat_epochs, freq = getSSVEPs(erps_days, epochs_days, epochs, settings, bids)
    fftdat_epochs = np.nanmean(fftdat_epochs,axis=0)  # average across trials to get the same shape for single trial SSVEPs

    # # get ssvep amplitudes
    SSVEPs_prepost_epochs, SSVEPs_prepost_channelmean_epochs, BEST_epochs = getSSVEPS_conditions(settings, fftdat_epochs, freq, cueduncued=0)  # single trial - cant do ERP using this method, SSVEPs average out

    # Plot SSVEP results
    ERPstring = 'Single Trial'
    plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs, settings, ERPstring, bids)

    # Topoplot SSVEPs
    topoinfo = topoplot_SSVEPs(raw, SSVEPs_prepost_epochs, ERPstring, settings, bids)

    # get wavelets
    wavelets_prepost = get_wavelets_prepost(epochs_days, settings, epochs, BEST_epochs, bids, cueduncued=0)

    np.savez(bids.direct_results / Path(bids.substring + "EEG_pre_post_results_coherentmotionepochs"),
             SSVEPs_prepost_channelmean_epochs=SSVEPs_prepost_channelmean_epochs,
             SSVEPs_prepost_epochs=SSVEPs_prepost_epochs,
             wavelets_prepost=wavelets_prepost,
             timepoints_zp=timepoints_zp,
             fftdat=fftdat, fftdat_epochs=fftdat_epochs, freq=freq, topoinfo=topoinfo)


def getSSVEPs(erps_days, epochs_days, epochs, settings, bids):
    from scipy.fft import fft, fftfreq, fftshift
    import matplotlib.pyplot as plt

    # 0 pad
    erps_days_use = erps_days.copy()
    epochs_days_use = epochs_days.copy()
    erps_days_use[:, epochs.times < 0, :, :, :, :] = 0
    erps_days_use[:, epochs.times > 1, :, :, :, :] = 0
    epochs_days_use[:, :, epochs.times < 0, :, :, :, :] = 0
    epochs_days_use[:, :, epochs.times > 1, :, :, :, :] = 0

    # for fft - fft
    fftdat = np.abs(fft(erps_days_use, axis=1)) / len(epochs.times)
    fftdat_epochs = np.abs(fft(epochs_days_use, axis=2)) / len(epochs.times)

    # plot single trial FFT spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.nanmean(np.mean(fftdat_epochs, axis=1), axis=0)

    for day_count in np.arange(2):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        axuse.plot(freq, chanmeanfft[:, 0, 0, 0, day_count].T, '-', label='Space/Left_diag', color=settings.lightteal)  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, 0, day_count].T, '-', label='Space/Right_diag', color=settings.medteal)  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, 0, day_count].T, '-', label='Feat/Black', color=settings.darkteal)  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, 0, day_count].T, '-', label='Feat/White', color=settings.yellow)  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, 1)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()

    titlestring = bids.substring + 'Single Trial FFT Spectrum move epochs'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    return fftdat, fftdat_epochs, freq


def getSSVEPS_conditions(settings, fftdat, freq, cueduncued):

    fftdatuse = fftdat[:,:,:,:,cueduncued,:]

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
                tmp = np.mean(np.mean(np.mean(fftdat[:, hz_attn_index[space_count, feat_count].astype(int), :, :, :, day_count], axis=1), axis=1), axis=1)# average across cue conditions to get best electrodes for frequencies
                BEST[:, space_count, feat_count, day_count] = tmp.argsort()[-settings.num_best:]
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

                        SSVEPs[cuetype, level, space_count, feat_count, day_count] = np.mean(fftdatuse[bestuse, hzuse, cuetype, level, day_count], axis=0)
                        SSVEPs_topo[:, cuetype, level, space_count, feat_count, day_count] = fftdatuse[:, hzuse, cuetype, level, day_count]

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

    titlestring = bids.substring + ' ' + ERPstring + ' SSVEPs pre-post motepochs'
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

    titlestring = bids.substring + ' ' + ERPstring + ' SSVEP selectivity pre-post  motepochs'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')


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


    titlestring = bids.substring + ' ' + ERPstring + ' Topoplots pre-post motepochs'
    fig.suptitle(titlestring)
    # plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    return topodat.info


def get_wavelets_prepost(epochs_days, settings, epochs, BEST, bids,cueduncued):
    import mne
    import matplotlib.pyplot as plt
    from pathlib import Path
    # epochs_days = np.empty((int(num_epochs), settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_attd_unattd, settings.num_days))
    epochs_days_tmp = epochs_days[:, :, :, :, :, cueduncued, :]
    # erps_days_wave = epochs_days.transpose(2, 3, 4, 0, 1 )  # [cuetype, level, day, chans,time,]

    # Get wavelets for each frequency
    num_epochs = 240
    wavelets_tmp = np.empty((settings.num_attnstates, settings.num_levels, int(num_epochs), len(epochs.times), settings.num_spaces, settings.num_features, settings.num_days))
    for space_count, space in enumerate(['Left_diag', 'Right_diag']):
        for feat_count, feat in enumerate(['Black', 'White']):
            for day_count, day_val in enumerate(settings.daysuse):
                for cuetype_count, cuetype in enumerate(['space', 'feat']):
                    bestuse = BEST[:, space_count, feat_count, day_count].astype(int)
                    hzuse = settings.hz_attn[space_count, feat_count]
                    datuse = np.mean(epochs_days_tmp[:, bestuse, :, cuetype_count, :, day_count], axis=0).transpose(2, 0, 1)
                    wavelets_tmp[cuetype_count, :, :, :, space_count, feat_count, day_count] = np.squeeze(mne.time_frequency.tfr_array_morlet(datuse, settings.samplingfreq, freqs=[hzuse],n_cycles=[hzuse], output='power'))
    wavelets = np.squeeze(np.nanmean(wavelets_tmp, axis=2))

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

            axuse.axvline(0,0,1000,linewidth=2, color='k')

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
            axuse.set_xlim(-1, 2)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn] + ' ' + settings.string_prepost[dayuse])

    titlestring = bids.substring + ' wavelets pre-post mot epochs'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    return wavelets_prepost


def collateEEGprepost(settings):
    print('collating SSVEP amplitudes pre Vs. post training')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_motionepochs, settings.samplingfreq)
    timelimits_data, timepoints, frequencypoints, zeropoint = helper.get_timing_variables(settings.timelimits_motionepochs, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_topodat_group = np.empty((settings.num_electrodes, settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints) +1, settings.num_attnstates, settings.num_levels, settings.num_attd_unattd, settings.num_days, num_subs))
    wavelets_prepost_group = np.empty((len(timepoints_zp) +1, settings.num_days, settings.num_attd_unattd, settings.num_attnstates, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_pre_post_results_coherentmotionepochs.npz"), allow_pickle=True)  #

        # store results
        SSVEPs_epochs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']
        SSVEPs_topodat_group[:, :, :, :, sub_count] = results['SSVEPs_prepost_epochs']
        fftdat_epochs_group[:, :, :, :, :, :, sub_count] = results['fftdat_epochs']
        wavelets_prepost_group[:, :, :, :, sub_count] = results['wavelets_prepost']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

        if sub_count == 1:
            # get EEG data
            raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count=0, day_val=1,settings=settings)

    np.savez(bids.direct_results_group / Path("EEGResults_prepost_motcoher"),
             SSVEPs_epochs_prepost_group=SSVEPs_epochs_prepost_group,
             SSVEPs_topodat_group=SSVEPs_topodat_group,
             fftdat_epochs_group=fftdat_epochs_group,
             wavelets_prepost_group=wavelets_prepost_group,
             timepoints_use=timepoints_use,
             freq=freq)

    # plot grand average frequency spectrum
    fftdat_epochs_ave = np.mean(fftdat_epochs_group[:, :, :, :, 0, :, :], axis=5)
    plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    plotGroupSSVEPsprepost(SSVEPs_epochs_prepost_group, bids, ERPstring='Single Trial', settings=settings)
    topoplot_SSVEPs_group(raw, SSVEPs_topodat_group, ERPstring='Single Trial', settings=settings, bids=bids)

    # plot wavelet results
    wavelets_prepost_group = wavelets_prepost_group / 3600
    wavelets_prepost_ave = np.mean(wavelets_prepost_group, axis=4)
    wavelets_prepost_std = np.std(wavelets_prepost_group, axis=4) / num_subs

    # plot wavelet data
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 15))
    for attn in np.arange(settings.num_attnstates):
        for dayuse in np.arange(settings.num_days):
            if (dayuse == 0): axuse = ax1[attn]
            if (dayuse == 1): axuse = ax2[attn]
            axuse.axvline(0, 0, 1000, linewidth=2, color='k')
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
            axuse.set_xlim(-0.5, 1.5)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.set_ylim(0.8, 1.2)
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn] + ' ' + settings.string_prepost[dayuse])

    titlestring = 'Group Mean wavelets pre-post mot coher' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    #plot difference waves
    diffwave = wavelets_prepost_group[:,:,0,:,:] - wavelets_prepost_group[:,:,1,:,:]
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

                axuse.fill_between(timepoints_use,wavelets_prepost_ave[:, dayuse, attn] - wavelets_prepost_std[:, dayuse, attn],wavelets_prepost_ave[:, dayuse, attn] + wavelets_prepost_std[:, dayuse, attn], alpha=0.3, facecolor=coluse[dayuse])
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, attn], color=coluse[dayuse], label=settings.string_prepost[dayuse])

            else:
                axuse.fill_between(timepoints_use, wavelets_prepost_ave[:, dayuse, attn] - wavelets_prepost_std[:, dayuse, attn],wavelets_prepost_ave[:, dayuse, attn] + wavelets_prepost_std[:, dayuse,  attn],alpha=0.3, facecolor=coluse[dayuse])
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse,  attn], color=coluse[dayuse],label=settings.string_prepost[dayuse])

            axuse.set_xlim(-0.5, 1.5)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.set_ylim(0, 0.2)
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn])

    titlestring = 'Group Mean wavelets pre-post diff mot coher' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # save attentional selectivity for stats - single trial
    ssvep_selectivity_prepost = SSVEPs_epochs_prepost_group[0, :, :, :] - SSVEPs_epochs_prepost_group[1, :, :, :]
    tmp = np.reshape(ssvep_selectivity_prepost,
                     (4, settings.num_subs))  # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost_epochs_motcoher.npy"), tmp)

    np.save(bids.direct_results_group / Path("group_ssvep_prepost_motcoher.npy"), SSVEPs_epochs_prepost_group)


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

    titlestring = ERPstring + ' Topoplots pre-post mot coher'
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

    titlestring = ERPstring + ' Topoplots pre-post Selectivity mot coher'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    ####### Difference Plot ( Attd - unattd)(Day4 - Day1)

    SSVEPs_Select_day = SSVEPs_Select[:, 1, :] - SSVEPs_Select[:, 0, :]  #  day, attd

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    vmin, vmax = -0.035, 0.035 #-np.max(np.abs(SSVEPs_Select_day[:])), np.max(np.abs(SSVEPs_Select_day[:]))  # get limits

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

    titlestring = ERPstring + ' Topoplots selectivity training efct mot coher'
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
        axuse.set_ylim(0.3, 0.9)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()
        axuse.set_frame_on(False)

    titlestring = 'Group Mean '+ ERPstring +' FFT Spectrum mot coher' + settings.string_attntrained[settings.attntrained]
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

    titlestring = 'Group Mean ' + ERPstring + ' SSVEPs pre-post TRAIN mot coher' + settings.string_attntrained[
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

    titlestring = 'Group Mean ' + ERPstring + ' SSVEP selectivity pre-post mot coher' + settings.string_attntrained[
        settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


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
        results = np.load(bids.direct_results_group / Path("EEGResults_prepost_motcoher.npz"), allow_pickle=True)  #

        SSVEPs_prepost_group = results['SSVEPs_epochs_prepost_group']  #[attd,day,cuetype,sub]
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
    df_selectivity.to_csv(bids.direct_results_group_compare / Path("motiondiscrim_SelectivityResults_ALL_motcoher.csv"),
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
        ax[i].set_ylim([0, 1.75])
        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task SSVEP Amplitudes by attention type mot coher'
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

    titlestring = 'Attentional Selectivity PrePost Compare Training  mot coher'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    ################# SELECTIVITY TOPO #################
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

    titlestring = 'Attentional Selectivity topo PrePost Compare Training mot coher'
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

    titlestring = 'Motion Task SSVEP Selectivity by Day pre Vs. post mot coher'
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
    df_SSVEPtraineffects["∆ Selectivity Topo"] = tmpd4['SSVEPs_var'] - tmpd1['SSVEPs_var']

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

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        if i == 0:
            ax[i].set_ylim([-0.75, 0.75])
        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task SSVEP Selectivity training effect by attention mot coher'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')


    # Selectivity Topo
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange, settings.red]

    for i in np.arange(2):
        datplot = df_SSVEPtraineffects[df_SSVEPtraineffects["Attention Type"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Trained", y="∆ Selectivity Topo", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Trained", y="∆ Selectivity Topo", data=datplot, palette=sns.color_palette(colors),
                       style="ticks",
                       ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)
        if i == 0:
            ax[i].set_ylim([-0.4, 0.6])
        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task SSVEP Selectivity topo training effect by attention mot coher'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')
