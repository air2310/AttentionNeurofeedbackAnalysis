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

       # average across trials
    # erps_days = np.squeeze(np.nanmean(epochs_days, axis=0))
    # erps_days_wave = np.squeeze(np.nanmean(epochs_days, axis=0))
    # timepoints_zp = epochs.times

    # get wavelets
    wavelets_prepost = get_wavelets_prepost(erps_days_wave, settings, epochs, BEST, bids)
