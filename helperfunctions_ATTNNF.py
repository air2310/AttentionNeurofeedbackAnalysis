import numpy as np
from pathlib import Path
import mne

# setup Classes and functions
class SetupMetaData:
    # Describe the options for settings
    string_attntrained = ["Space", "Feature"]
    string_tasknames =  ['AttnNFMotion', 'AttnNFVisualSearch', 'AttnNFnback']
    string_testday = ['Day 1', 'Day 2', 'Day 3', 'Day 4']
    string_prepost = ['pretest', 'posttest'];
    string_cuetype  = ["Space", "Feature"]
    string_testtrain = ["Test", "Train"]
    string_attd_unattd = ["Attended", "UnAttended"]

    # Flicker frequencies
    hz_attn = np.array([[8, 12], [4.5, 14.4]]) #['\B \W'], ['/B /W']

    # numerical settings
    num_features = 2
    num_spaces = 2
    num_attnstates = 2 # space, feature
    num_levels = 2 # left diag, right diag OR black, white
    num_best = 4 # number of electrodes to average SSVEPs over
    num_attd_unattd = 2

    # Directories
    direct_dataroot = Path("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Data/")
    direct_resultsroot = Path("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/")

    # relevant triggers
    trig_cuestart_cuediff = np.array(([121, 122], [123, 124]))

    # initialise
    def __init__(self, attntrained):
        self.attntrained = attntrained

        # get correct subject indices
        if (self.attntrained == 0): # Space
            self.subsIDX = np.array(([10]))
        else: # Feature
            self.subsIDX = np.array(([1]))
        self.num_subs = len(self.subsIDX)


    def get_settings_EEG_prepost(self):
        # Task Settings
        self.testtrain = 0 # 0 = test, 1 = train
        self.task = 0 # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        self.daysuse = [1, 4]
        self.num_trials = 192
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

    # load EEG file
    raw = mne.io.read_raw_brainvision(file2use, preload=True, scale=1e6)
    print(raw.info)

    # pick events
    events = mne.find_events(raw, stim_channel="TRIG")
    mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp)
    print('Found %s events, first five:' % len(events))
    print(events[:5])

    return raw, events

def getSSVEPs(erps_days, epochs_days, epochs, settings):
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

    fig, (ax1, ax2) = plt.subplots(1, 2)
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.mean(fftdat, axis=0)

    for day_count in np.arange(2):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Space/Left_diag')  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Space/Right_diag')  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='Feat/Black')  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='Feat/White')  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, .7)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()

    fig.suptitle('ERP FFT Spectrum')

    # plot single trial FFT spectrum
    fig, (ax1, ax2) = plt.subplots(1, 2)
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins
    chanmeanfft = np.nanmean(np.mean(fftdat_epochs, axis=1), axis=0)

    for day_count in np.arange(2):
        if (day_count == 0): axuse = ax1
        if (day_count == 1): axuse = ax2
        axuse.plot(freq, chanmeanfft[:, 0, 0, day_count].T, '-', label='Space/Left_diag')  # 'Space/Left_diag'
        axuse.plot(freq, chanmeanfft[:, 0, 1, day_count].T, '-', label='Space/Right_diag')  # 'Space/Right_diag'
        axuse.plot(freq, chanmeanfft[:, 1, 0, day_count].T, '-', label='Feat/Black')  # 'Feat/Black'
        axuse.plot(freq, chanmeanfft[:, 1, 1, day_count].T, '-', label='Feat/White')  # 'Feat/White'

        axuse.set_xlim(2, 20)
        axuse.set_ylim(0, 1)
        axuse.set_title(settings.string_prepost[day_count])
        axuse.legend()

    fig.suptitle('single Trial FFT Spectrum')

    return fftdat, fftdat_epochs