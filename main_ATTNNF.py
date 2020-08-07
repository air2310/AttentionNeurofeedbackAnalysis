# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, fftshift

# setup Classes and functions
class SetupMetaData:
    # Describe the options for settings
    string_attntrained = ["Feature", "Space"]
    string_tasknames =  ['AttnNFMotion', 'AttnNFVisualSearch', 'AttnNFnback']
    string_testday = ['Day 1', 'Day 2', 'Day 3', 'Day 4']
    string_prepost = ['pretest', 'posttest'];
    string_cuetype  = ["Feature", "Space"]
    string_testtrain = ["Test", "Train"]

    # Flicker frequencies
    hz_attn = np.array([[8, 12], [4.5, 14.4]]) #['\B \W'], ['/B /W']

    # numerical settings
    num_features = 2
    num_spaces = 2
    num_attnstates = 2 # space, feature
    num_levels = 2 # left diag, right diag OR black, white
    num_best = 4 # number of electrodes to average SSVEPs over

    # Directories
    direct_dataroot = Path("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Data/")
    direct_resultsroot = Path("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/")

    # relevant triggers
    trig_cuestart_cuediff = np.array(([121, 122], [123, 124]))

    # initialise
    def __init__(self, attntrained):
        self.attntrained = attntrained

        # get correct subject indices
        if (self.attntrained == 0): # Feature
            self.subsIDX = np.array(([1]))
        else: # Space
            self.subsIDX = np.array(([10]))
        self.num_subs = len(self.subsIDX)


    def get_settings_EEG_prepost(self):
        # Task Settings
        self.testtrain = 0 # 0 = test, 1 = train
        self.task = 0 # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        self.daysuse = [1, 4]
        # self.daysuse = [1]

        # EEG settings
        self.samplingfreq = 1200
        self.num_electrodes = 9

        # Timing Settings
        self.timelimits = np.array([0, 6]) # Epoch start and end time (seconds) relative to cue onset
        self.zeropading = 2
        self.timelimits_zeropad = np.array([self.timelimits[0]-self.zeropading, self.timelimits[1]+self.zeropading])
        return self


class BIDS_FileNaming:
    def __init__(self, subject_idx, settings, day_val):
        if (subject_idx < 10):
            self.substring = 'sub-0' + str(subject_idx)
        else:
            self.substring = 'sub-' + str(subject_idx)

        # case string: describes the current condition - for filenames
        self.casestring = self.substring + '_task-' + settings.string_tasknames[settings.task] + '_day-' + str(day_val) + '_phase-' + settings.string_testtrain[settings.testtrain]

        # directories where things are stored
        self.direct_data_eeg = settings.direct_dataroot / Path('Train' + settings.string_attntrained[settings.attntrained] + '/' + self.substring + '/eeg/')
        self.direct_data_behave = settings.direct_dataroot / Path('Train' + settings.string_attntrained[settings.attntrained] + '/' + self.substring + '/behave/')
        self.direct_results =  settings.direct_resultsroot / Path('Train' + settings.string_attntrained[settings.attntrained] + '/' + self.substring + '/')

        # filenames
        self.filename_eeg = self.casestring + '_eeg'
        self.filename_chan = self.casestring + '_channels*'
        self.filename_evt = self.casestring + '_events*'
        self.filename_behave = self.casestring + '_behav*'
        self.filename_vissearch = self.casestring + '_vissearch*'
        self.filename_nback = self.casestring + 'nback*'

def get_timing_variables(timelimits,samplingfreq):
    timelimits_data = timelimits * samplingfreq

    num_seconds = timelimits[1] - timelimits[0]
    num_datapoints =  timelimits_data[1] -  timelimits_data[0]

    timepoints = np.arange(timelimits[0], timelimits[1] , 1 / samplingfreq) # Time (in seconds) of each sample in epoch relative to trigger
    frequencypoints = np.arange(0, samplingfreq , 1 / num_seconds) # frequency (in Hz) of each sample in epoch after fourier transform

    zeropoint = np.argmin(np.abs(timepoints - 0))
    return timelimits_data, timepoints, frequencypoints, zeropoint

def get_eeg_data(bids):
    # decide which EEG file to use
    possiblefiles = []
    filesizes = []
    for filesfound in bids.direct_data_eeg.glob(bids.filename_eeg + "*.vhdr"):
        filesizes.append(filesfound.stat().st_size)
        possiblefiles.append(filesfound)

    file2useIDX = np.argmax(filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
    file2use = possiblefiles[file2useIDX]

    # load EEG file
    raw = mne.io.read_raw_brainvision(file2use, preload=True, scale=1e6)
    print(raw.info)

    # pick events
    events = mne.find_events(raw, stim_channel="TRIG")
    mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp)
    print('Found %s events, first five:' % len(events))
    print(events[:5])

    return raw, events

# setup generic settings
attntrained = 0 # ["Feature", "Space"]
settings = SetupMetaData(attntrained)

print("Analysing Data for condition train: " + settings.string_attntrained[settings.attntrained])

# iterate through subjects
# for sub_count, sub_val in enumerate(settings.subsIDX):
sub_count = 0
sub_val = 1

# decide whether to analyse EEG Pre Vs. Post Training
analyseEEGprepost = True
if (analyseEEGprepost):
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # get timing settings
    # timelimits_data, timepoints, frequencypoints, zeropoint = get_timing_variables(settings.timelimits,settings.samplingfreq)
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = get_timing_variables(settings.timelimits_zeropad,settings.samplingfreq)


    # preallocate
    erps_days = np.empty(( settings.num_electrodes, len(timepoints_zp) + 1, settings.num_levels, settings.num_attnstates,
                          settings.num_days))
    ffts_days = np.empty(( settings.num_electrodes, len(timepoints_zp) + 1, settings.num_levels, settings.num_attnstates,
                          settings.num_days))
    erps_days[:] = np.nan
    ffts_days[:] = np.nan

    # iterate through test days to get data
    for day_count, day_val in enumerate(settings.daysuse):
        # get file names
        bids = BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # get EEG data
        raw, events = get_eeg_data(bids)

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

        # for fft - get data for each  condition
        erps_days[:, :, 0, 0, day_count] = np.squeeze(np.mean(epochs['Space/Left_diag'].get_data(), axis=0))
        erps_days[:, :, 1, 0, day_count] = np.squeeze(np.mean(epochs['Space/Right_diag'].get_data(), axis=0))
        erps_days[:, :, 0, 1, day_count] = np.squeeze(np.mean(epochs['Feat/Black'].get_data(), axis=0))
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
    spaceSSVEPs = np.empty((settings.num_electrodes, settings.num_days, settings.num_attnstates, settings.num_levels))
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
    featureSSVEPs = np.empty((settings.num_electrodes, settings.num_days, settings.num_attnstates, settings.num_levels))
    for level_count, level in enumerate(['Black', 'White']):
        if (level == 'Black'):
            attendedSSVEPs =   np.mean(fftdat[:, hz_attn_index[:, black].astype(int), cuetype, level_count, :], axis = 1) # average across black frequencies at both spatial positions
            unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, white].astype(int), cuetype, level_count, :], axis = 1) # average across white frequencies at both spatial positions

        if (level == 'White'):
            attendedSSVEPs =   np.mean(fftdat[:, hz_attn_index[:, white].astype(int), cuetype, level_count, :], axis = 1) # average across white frequencies at both spatial positions
            unattendedSSVEPs = np.mean(fftdat[:, hz_attn_index[:, black].astype(int), cuetype, level_count, :], axis = 1) # average across black frequencies at both spatial positions

        featureSSVEPs[:, :, 0, level_count] = attendedSSVEPs
        featureSSVEPs[:, :, 1, level_count] = unattendedSSVEPs


    # average across cue types and store the SSVEPs alltogether for plotting and further analysis