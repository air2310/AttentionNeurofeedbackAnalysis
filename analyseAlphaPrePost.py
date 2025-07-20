import numpy as np
from pathlib import Path
import mne
import pandas as pd

import helperfunctions_ATTNNF as helper

import fooof
import matplotlib.pyplot as plt
from collections import defaultdict
from fooof import FOOOF
from fooof.utils import trim_spectrum, interpolate_spectrum

def analyseprepost(settings, sub_val):
    print('analysing Alpha amplitudes pre Vs. post training')

    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # iterate through test days to get data
    alpha_results = defaultdict(list)
    alpha_spectrums = defaultdict(list)
    for day_count, day_val in enumerate(settings.daysuse):

        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # get EEG data
        try:
            raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count, day_val, settings)

            # Epoch to events of interest
            event_id = {'Feat/Black': 121, 'Feat/White': 122,
                        'Space/Left_diag': 123, 'Space/Right_diag': 124}  # will be different triggers for training days

            epochs = mne.Epochs(eeg_data_interp, events, event_id=event_id, tmin=settings.timelimits[0],
                                tmax=settings.timelimits[1],
                                baseline=(0, 1 / settings.samplingfreq), reject=dict(eeg=250), detrend=1)  #
            epochs.drop_bad()

        except:
            print('File Missing')
            break

        # Apply FFT
        PSDS = {}
        for cond in ['Space', 'Feat']:
            spectrum = epochs[cond].compute_psd("welch", n_fft=len(epochs.times), n_overlap=0,
                                                    n_per_seg=len(epochs.times), fmin=1, fmax=30,
                                                    window="boxcar", verbose=False)
            psds, freqs = spectrum.get_data(return_freqs=True)
            PSDS[cond] = np.squeeze(np.mean(psds,1))
        PSDS['freqs'] = freqs

        # Store spectrums
        for cond in ['Space', 'Feat']:
            alpha_spectrums['Freqs'].extend(PSDS['freqs'])
            alpha_spectrums['Amp'].extend(PSDS[cond].mean(axis=0))
            alpha_spectrums['Attention_Type'].extend([cond]*len(freqs))

            alpha_spectrums['TestDay'].extend([settings.string_prepost[day_count]]*len(freqs))
            alpha_spectrums['SubID'].extend([bids.substring]*len(freqs))
            alpha_spectrums['AttentionTrained'].extend([settings.string_attntrained[settings.attntrained]]*len(freqs))

        # show
        plt.figure()
        plt.plot(PSDS['freqs'], PSDS['Space'].mean(axis=0))
        plt.plot(PSDS['freqs'], PSDS['Feat'].mean(axis=0))
        plt.xlim([2, 40])
        plt.legend(['Space', 'Feat'])

        # Run FOOOF
        # Set the frequency range to fit the model
        freq_range = [2, 40]

        # Setup interpolations

        tmp = settings.hz_attn.flatten()
        interp_freqs = tmp.tolist() + (tmp*2).tolist() + [18] # + (tmp*3).tolist()
        interp_ranges = [[freq-.5, freq+.5] for freq in interp_freqs]

        # cycle through conds
        for cond in ['Space', 'Feat']:
            # Initialize a FOOOF object
            fm = FOOOF()
            # Interpolate over tags
            freqs_int2, powers_int2 = interpolate_spectrum(PSDS['freqs'], PSDS[cond].mean(axis=0), interp_ranges)

            # Fit spectrum
            # fm.fit(PSDS['freqs'], PSDS[cond].mean(axis=0), freq_range)
            fm.fit(freqs_int2, powers_int2, freq_range)

            # Check results
            fm.plot(plot_peaks='shade')
            fm.print_results()
            ap_params, peak_params, r_squared, fit_error, gauss_params = fm.get_results() # apparams  (offset, exponent):

            # Classify Alpha
            possibleidx = np.where((peak_params[:,0] >8) & (peak_params[:,0] <12 ))[0]
            if len(possibleidx)>0:
                # pick the widest component centred in the alpha range
                idxalpha=possibleidx[np.argmax(peak_params[possibleidx, 2])]
                CF, PW, BW = peak_params[idxalpha,:]
            else:
                print('No alpha paramaters identidied!')
                CF, PW, BW = np.nan, np.nan, np.nan

            # Store
            alpha_results['Centre Freq.'].append(CF)
            alpha_results['Alpha Peak'].append(PW)
            alpha_results['Alpha Band Width'].append(BW)
            alpha_results['FOOOF Offset'].append(ap_params[0])
            alpha_results['FOOOF Exponent'].append(ap_params[1])
            alpha_results['FOOOF rsquared'].append(r_squared)

            alpha_results['Attention_Type'].append(cond)
            alpha_results['TestDay'].append(settings.string_prepost[day_count])
            alpha_results['SubID'].append(bids.substring)
            alpha_results['AttentionTrained'].append(settings.string_attntrained[settings.attntrained])


    # Save
    alpha_spectrums_pd = pd.DataFrame(alpha_spectrums)
    filename = bids.direct_results / Path(bids.substring + "EEG_alphaspectrum_results.pkl")
    alpha_spectrums_pd.to_pickle(filename)

    alpha_results_pd = pd.DataFrame(alpha_results)
    filename = bids.direct_results / Path(bids.substring + "EEG_alpha_results.pkl")
    alpha_results_pd.to_pickle(filename)


def CollateEEGprepost(settings, sub_val):
    print('Collating Alpha amplitudes pre Vs. post training')

    alpha_spectrums_all = pd.DataFrame()
    alpha_results_all = pd.DataFrame()
    # get settings specific to this analysis
    for attntrained in range(3):  # 0 = Space, 1 = Feature, 2 = Sham
        settings = helper.SetupMetaData(attntrained)

        # get settings specific to this analysis
        settings = settings.get_settings_EEG_prepost()

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDX):
            bids = helper.BIDS_FileNaming(sub_val, settings, day_val=1)
            print(bids.casestring)

            # Load data
            # Save
            filename = bids.direct_results / Path(bids.substring + "EEG_alphaspectrum_results.pkl")
            alpha_spectrums_pd = pd.read_pickle(filename)
            alpha_spectrums_all = pd.concat((alpha_spectrums_all, alpha_spectrums_pd))

            filename = bids.direct_results / Path(bids.substring + "EEG_alpha_results.pkl")
            alpha_results_pd = pd.read_pickle(filename)
            alpha_results_all = pd.concat((alpha_results_all, alpha_results_pd))

    # Fill nans
    alpha_results_all.loc[np.isnan(alpha_results_all['Alpha Peak']),'Alpha Peak'] = 0
    alpha_results_all = alpha_results_all.loc[alpha_results_all['FOOOF rsquared']>.95, :]
    # Show results!
    import seaborn as sns
    for var in ['Alpha Peak', 'FOOOF Offset', 'FOOOF Exponent', 'FOOOF rsquared']:
        plt.figure()
        sns.boxplot(alpha_results_all, y=var, x='TestDay', hue='Attention_Type')
        plt.figure()
        sns.boxplot(alpha_results_all, y=var, x='TestDay', hue='AttentionTrained')


    tmp = alpha_results_all.loc[alpha_results_all['TestDay']=='post-training', 'Alpha Peak'].mean() - alpha_results_all.loc[alpha_results_all['TestDay']=='pre-training', 'Alpha Peak'].mean()..values
    # compute training differences and then export for analysis in r.
