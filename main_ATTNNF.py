# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt
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
        num_epochs = (settings.num_trials) / 4
        epochs_days = np.empty(( int(num_epochs), settings.num_electrodes, len(timepoints_zp) + 1,  settings.num_attnstates, settings.num_levels,
                              settings.num_days))

        epochs_days[:] = np.nan

        # iterate through test days to get data
        for day_count, day_val in enumerate(settings.daysuse):
            # TODO: save daily plots for events, drop logs, ERPs and FFTs
            # get file names
            bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
            print(bids.casestring)

            # get EEG data
            raw, events = helper.get_eeg_data(bids)

            # Filter Data
            raw.filter(l_freq=1, h_freq=45, h_trans_bandwidth=0.1)

            # Epoch to events of interest
            event_id = {'Feat/Black': 121, 'Feat/White': 122,
                        'Space/Left_diag': 123, 'Space/Right_diag': 124} # will be different triggers for training days

            epochs = mne.Epochs(raw, events, event_id=event_id, tmin=settings.timelimits_zeropad[0], tmax=settings.timelimits_zeropad[1],
                                baseline=(0, 1 / settings.samplingfreq), picks=np.arange(settings.num_electrodes),
                                reject=dict(eeg=400), detrend=1) #

            # drop bad channels
            epochs.drop_bad()
            epochs.plot_drop_log()
            # epochs2 = epochs.equalize_event_counts(event_id, method='mintime')

            # get data for each  condition
            epochs_days[0:sum(elem == [] for elem in epochs['Space/Left_diag'].drop_log), :, :, 0, 0, day_count] = epochs['Space/Left_diag'].get_data()
            epochs_days[0:sum(elem == [] for elem in epochs['Space/Right_diag'].drop_log), :, :, 0, 1, day_count] = epochs['Space/Right_diag'].get_data()
            epochs_days[0:sum(elem == [] for elem in epochs['Feat/Black'].drop_log), :, :, 1, 0, day_count] = epochs['Feat/Black'].get_data()
            epochs_days[0:sum(elem == [] for elem in epochs['Feat/White'].drop_log), :, :, 1, 1, day_count] = epochs['Feat/White'].get_data()

        # average across trials
        erps_days = np.squeeze(np.nanmean(epochs_days, axis=0))
        erps_days_wave = np.squeeze(np.nanmean(epochs_days, axis=0))
        timepoints_zp = epochs.times

        # Get SSVEPs
        fftdat, fftdat_epochs, freq = helper.getSSVEPs(erps_days, epochs_days, epochs, settings, bids)
        fftdat_epochs = np.nanmean(fftdat_epochs, axis=0) # average across trials to get the same shape

        SSVEPs_prepost, SSVEPs_prepost_channelmean, BEST = helper.getSSVEPS_conditions(settings, fftdat, freq)
        SSVEPs_prepost_epochs, SSVEPs_prepost_channelmean_epochs, BEST_epochs = helper.getSSVEPS_conditions(settings, fftdat_epochs, freq)

        # get wavelets
        wavelets_prepost = helper.get_wavelets_prepost(erps_days_wave, settings, epochs,BEST, bids)

        # plot topos
        # TODO: add topoplotting

        # Plot SSVEP results
        ERPstring = 'ERP'
        helper.plotResultsPrePost_subjects(SSVEPs_prepost_channelmean, settings, ERPstring, bids)

        ERPstring = 'Single Trial'
        helper.plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs, settings, ERPstring, bids)
        # when we return - check single trial SSVEP amplitudes. figure out if this script is wrong or if the matlab script is.

        np.savez(bids.direct_results / Path(bids.substring + "EEG_pre_post_results.npy"), SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)