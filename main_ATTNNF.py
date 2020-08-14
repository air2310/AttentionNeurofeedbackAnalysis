# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt
import Analysis_Code.helperfunctions_ATTNNF as helper
import Analysis_Code.functions_getEEGprepost as geegpp

# notes
# integrate behavioural analyses with python
# find a way to normalise day by day SSVEP amplitudes? # use signal to noise instead?!
# correlate behaviour with ssvep selectivity
# look at differences between classifiable and unclassifiable peeps.
# check prepost differences in nback and visual search tasks
# add errorbars and standard ylims to wavelet plots
# add subject scatterpoints to behavioural results
# switch to within subjects error across plots
# analyse behaviour during training - across the three days.
# figure out topoplotting

# Decide which analyses to do
analyseEEGprepost = False # analyse EEG Pre Vs. Post Training
collateEEGprepost = True # Collate EEG Pre Vs. Post Training across subjects

# setup generic settings
attntrained = 0 # ["Space", "Feature"]
settings = helper.SetupMetaData(attntrained)

print("Analysing Data for condition train: " + settings.string_attntrained[settings.attntrained])

# iterate through subjects for individual subject analyses
for sub_count, sub_val in enumerate(settings.subsIDX):

    if (analyseEEGprepost):
        print('analysing SSVEP amplitudes pre Vs. post training')

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
            raw, events, eeg_data_interp = geegpp.get_eeg_data(bids, day_count)

            # Epoch to events of interest
            event_id = {'Feat/Black': 121, 'Feat/White': 122,
                        'Space/Left_diag': 123, 'Space/Right_diag': 124} # will be different triggers for training days

            epochs = mne.Epochs(eeg_data_interp, events, event_id=event_id, tmin=settings.timelimits_zeropad[0], tmax=settings.timelimits_zeropad[1],
                                baseline=(0, 1 / settings.samplingfreq), picks=np.arange(settings.num_electrodes),
                                reject=dict(eeg=400), detrend=1) #

            # drop bad channels
            epochs.drop_bad()
            epochs.plot_drop_log()
            # epochs2 = epochs.equalize_event_counts(event_id, method='mintime')

            # visualise topo
            # epochs.plot_psd_topomap()

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
        fftdat, fftdat_epochs, freq = geegpp.getSSVEPs(erps_days, epochs_days, epochs, settings, bids)
        fftdat_epochs = np.nanmean(fftdat_epochs, axis=0) # average across trials to get the same shape for single trial SSVEPs

        SSVEPs_prepost, SSVEPs_prepost_channelmean, BEST = geegpp.getSSVEPS_conditions(settings, fftdat, freq) # trial average
        SSVEPs_prepost_epochs, SSVEPs_prepost_channelmean_epochs, BEST_epochs = geegpp.getSSVEPS_conditions(settings, fftdat_epochs, freq) # single trial

        # get wavelets
        wavelets_prepost = geegpp.get_wavelets_prepost(erps_days_wave, settings, epochs,BEST, bids)

        # plot topos
        # TODO: add topoplotting

        # Plot SSVEP results
        ERPstring = 'ERP'
        geegpp.plotResultsPrePost_subjects(SSVEPs_prepost_channelmean, settings, ERPstring, bids)

        ERPstring = 'Single Trial'
        geegpp.plotResultsPrePost_subjects(SSVEPs_prepost_channelmean_epochs, settings, ERPstring, bids)
        # TODO: figure out if this script is wrong or if the matlab script is.

        np.savez(bids.direct_results / Path(bids.substring + "EEG_pre_post_results"), SSVEPs_prepost_channelmean=SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs=SSVEPs_prepost_channelmean_epochs, wavelets_prepost=wavelets_prepost, timepoints_zp=timepoints_zp, erps_days_wave=erps_days_wave, fftdat=fftdat, fftdat_epochs=fftdat_epochs, freq=freq)

# Collate EEG prepost
if (collateEEGprepost):
    print('collating SSVEP amplitudes pre Vs. post training')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    fftdat_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    wavelets_prepost_group = np.empty((len(timepoints_zp) + 1, settings.num_days, settings.num_attd_unattd, settings.num_attnstates, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(sub_val, settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_pre_post_results.npz"), allow_pickle=True) #
        #saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

        # store results
        SSVEPs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean']
        SSVEPs_epochs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']
        fftdat_group[:,:,:,:,:,sub_count] = results['fftdat']
        fftdat_epochs_group[:, :, :, :, :, sub_count] = results['fftdat_epochs']
        wavelets_prepost_group[:,:,:,:,sub_count] = results['wavelets_prepost']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

    # plot grand average frequency spectrum
    fftdat_ave = np.mean(fftdat_group, axis = 5)
    geegpp.plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring='ERP', settings=settings, freq=freq)

    fftdat_epochs_ave = np.mean(fftdat_epochs_group, axis = 5)
    geegpp.plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    geegpp.plotGroupSSVEPsprepost(SSVEPs_prepost_group, bids, ERPstring='ERP', settings=settings)
    geegpp.plotGroupSSVEPsprepost(SSVEPs_epochs_prepost_group, bids, ERPstring='Single Trial', settings=settings)

    # plot wavelet results
    wavelets_prepost_ave = np.mean(wavelets_prepost_group, axis=4)
    wavelets_prepost_std = np.std(wavelets_prepost_group, axis=4)/num_subs

    # plot wavelet data
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 15))
    for attn in np.arange(settings.num_attnstates):
        for dayuse in np.arange(settings.num_days):
            if (dayuse == 0): axuse = ax1[attn]
            if (dayuse == 1): axuse = ax2[attn]
            if (attn == 0):
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 0, attn], color=settings.medteal,
                           label=settings.string_attd_unattd[0])
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 1, attn], color=settings.lightteal,
                           label=settings.string_attd_unattd[1])
            else:
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 0, attn], color=settings.orange,
                           label=settings.string_attd_unattd[0])
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 1, attn], color=settings.yellow,
                           label=settings.string_attd_unattd[1])
            axuse.set_xlim(-1, 6)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn] + ' ' + settings.string_prepost[dayuse])

    titlestring = 'Group Mean wavelets pre-post ' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # save attentional selectivity for stats
    ssvep_selectivity_prepost = SSVEPs_prepost_group[0, :, :, :] - SSVEPs_prepost_group[1, :, :, :]
    # day, attntype

    tmp = np.reshape(ssvep_selectivity_prepost, (4,settings.num_subs)) # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost.npy"), tmp)
    # TODO: get wserror
    # plot error around wavelets






