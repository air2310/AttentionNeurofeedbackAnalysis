import numpy as np
from pathlib import Path
import mne

# setup Classes and functions
class SetupMetaData:
    # Describe the options for settings
    string_attntrained = ["Space", "Feature"]
    string_tasknames =  ['AttnNFMotion', 'AttnNFVisualSearch', 'AttnNFnback']
    string_testday = ['Day 1', 'Day 2', 'Day 3', 'Day 4']
    string_prepost = ['pre-training', 'post-training'];
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


    # colours
    darkteal = [18 / 255, 47 / 255, 65 / 255]
    medteal = [5 / 255, 133 / 255, 134 / 255]
    lightteal = [78 / 255, 185 / 255, 159 / 255]
    yellow = [242 / 255, 176 / 255, 53 / 255]
    orange = [236 / 255, 85 / 255, 58 / 255]
    red = [193 / 255, 31 / 255, 58 / 255]

    # initialise
    def __init__(self, attntrained):
        self.attntrained = attntrained

        # get correct subject indices
        if (self.attntrained == 0): # Space
            self.subsIDX = np.array(([ 10, 11, 19, 22, 28, 29,38, 43, 45, 46, 49, 52, 53, 54, 59, 60, 64, 71, 74, 79, 81, 84 ]))
            self.subsIDXcollate = np.array(([10, 11, 19, 22, 28, 29,38, 43, 45, 46, 49, 52, 53, 54, 59, 60, 64, 71, 74, 79, 81, 84 ])) #, 19, 22, 28, 29, 43, 45, 46, 49, 52, 53, 54, 59, 60]))
            self.subsIDXall = np.array(([10, 11, 19, 22, 28, 29,38, 43, 45, 46, 49, 52, 53, 54, 59, 60, 64, 71, 74, 79, 81, 84]))

        else: # Feature
            self.subsIDX = np.array(([ 21, 23, 41, 47, 57, 58,63, 66, 67,68, 69, 70, 72, 73, 76, 77, 78, 80])) # 1, 2,
            self.subsIDXcollate = np.array(([1, 2, 4, 8, 9, 18, 21, 23, 41, 47, 57, 58,63, 66, 67,68, 69, 70, 72, 73, 76, 77, 78, 80 ])) #np.array(([1, 2, 4, 8, 9, 18, 21, 23, 41, 47, 57, 58,63, 66, 67,68, 69 ]))
            self.subsIDXall = np.array(([1, 2, 4, 8, 9, 18, 21, 23, 41, 47, 57, 58, 63, 66, 67, 68, 69, 70, 72, 73, 76, 77, 78, 80]))
            # 21 day 1 train files missing
        self.num_subs = len(self.subsIDXcollate)


    def get_settings_EEG_prepost(self): # settings specific to pre vs post training EEG analysis
        # Task Settings
        self.testtrain = 0 # 0 = test, 1 = train
        self.task = 0 # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        self.daysuse = [1, 4]
        self.num_trials = 192
        self.num_conditions = 4

        # EEG settings
        self.samplingfreq = 1200
        self.num_electrodes = 9

        # Timing Settings
        self.timelimits = np.array([0, 6]) # Epoch start and end time (seconds) relative to cue onset
        self.zeropading = 2
        self.timelimits_zeropad = np.array([self.timelimits[0]-self.zeropading, self.timelimits[1]+self.zeropading])

        # relevant triggers
        trig_cuestart_cuediff = {'Feat/Black': 121, 'Feat/White': 122,
                                 'Space/Left_diag': 123,
                                 'Space/Right_diag': 124}  # will be different triggers for training days

        return self

    def get_settings_EEG_duringNF(self): # settings specific to pre vs post training EEG analysis
        # Task Settings
        self.testtrain = 1 # 0 = test, 1 = train
        self.task = 0 # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 3
        # self.daysuse = [1]
        self.daysuse = [1, 2, 3]
        self.num_trials = 256
        self.num_conditions = 4

        # EEG settings
        self.samplingfreq = 1200
        self.num_electrodes = 9

        # Timing Settings
        self.timelimits = np.array([0, 6]) # Epoch start and end time (seconds) relative to cue onset
        self.zeropading = 2
        self.timelimits_zeropad = np.array([self.timelimits[0]-self.zeropading, self.timelimits[1]+self.zeropading])
        return self

    def get_settings_visualsearchtask(self):  # settings specific to pre vs post training EEG analysis
        self.testtrain = 0  # 0 = test, 1 = train
        self.task = 1  # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        self.daysuse = [1, 4]

        self.num_trialscond = 80
        self.num_setsizes = 3
        self.string_setsize = ["SS8", "SS12", "SS16"]

        return self

    def get_settings_nbacktask(self):  # settings specific to pre vs post training EEG analysis
        self.testtrain = 0  # 0 = test, 1 = train
        self.task = 2  # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        self.daysuse = [1, 4]

        self.num_trialsblock = 48
        self.num_blocks= 4
        self.num_trials = self.num_trialsblock * self.num_blocks

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
        self.direct_results.mkdir(parents=True, exist_ok=True)

        # group results
        self.direct_results_group = settings.direct_resultsroot / Path(
            'Train' + settings.string_attntrained[settings.attntrained] + '/group/')
        self.direct_results_group.mkdir(parents=True, exist_ok=True)

        # group results between subjects
        self.direct_results_group_compare = settings.direct_resultsroot / Path(
            'CompareSpaceFeat/group/')
        self.direct_results_group_compare.mkdir(parents=True, exist_ok=True)

        # filenames
        self.filename_eeg = self.casestring + '_eeg'
        self.filename_chan = self.casestring + '_channels'
        self.filename_evt = self.casestring + '_events'
        self.filename_behave = self.casestring + '_behav'
        self.filename_vissearch = self.casestring + '_vissearch'
        self.filename_nback = self.casestring + '_behav'

def get_timing_variables(timelimits,samplingfreq):
    timelimits_data = timelimits * samplingfreq

    num_seconds = timelimits[1] - timelimits[0]
    num_datapoints =  timelimits_data[1] -  timelimits_data[0]

    timepoints = np.arange(timelimits[0], timelimits[1] , 1 / samplingfreq) # Time (in seconds) of each sample in epoch relative to trigger
    frequencypoints = np.arange(0, samplingfreq , 1 / num_seconds) # frequency (in Hz) of each sample in epoch after fourier transform

    zeropoint = np.argmin(np.abs(timepoints - 0))
    return timelimits_data, timepoints, frequencypoints, zeropoint


def get_eeg_data(bids, day_count, settings):
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

    montage = {'Iz':  [0, -110, -40],
               'Oz': [0, -105, -15],
               'POz': [0,   -100, 15],
               'O1': [-40, -106, -15],
               'O2':  [40, -106, -15],
               'PO3': [-35, -101, 10],
               'PO4': [35,  -101, 10],
               'PO7': [-70, -110, 0],
               'PO8': [70, -110, 0] }

    montageuse = mne.channels.make_dig_montage(ch_pos=montage, lpa=[-82.5, -19.2, -46], nasion=[0, 83.2, -38.3], rpa=[82.2, -19.2, -46]) # based on mne help file on setting 10-20 montage

    # load EEG file
    raw = mne.io.read_raw_brainvision(file2use, preload=True, scale=1e6)
    print(raw.info)

    # raw.plot(remove_dc = False, scalings=dict(eeg=50))

    # pick events
    if (bids.substring == 'sub-68'):
        events = mne.find_events(raw, stim_channel="TRIG", min_duration=4/raw.info['sfreq'])
    else:
        events = mne.find_events(raw, stim_channel="TRIG")

    mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp)
    print('Found %s events, first five:' % len(events))
    print(events[:5])

    # set bad chans
    if (np.logical_and(bids.substring == 'sub-02', day_count == 0)):         raw.info['bads'] = ['O1']
    if (np.logical_and(bids.substring == 'sub-09', day_count == 0)):        raw.info['bads'] = ['PO4']
    if (np.logical_and(bids.substring == 'sub-23', day_count == 0)):        raw.info['bads'] = ['Iz']
    if (np.logical_and(bids.substring == 'sub-47', day_count == 0)):        raw.info['bads'] = ['PO8']
    if (np.logical_and(bids.substring == 'sub-53', day_count == 0)):        raw.info['bads'] = ['Iz']
    if (np.logical_and(bids.substring == 'sub-70', day_count == 0)):        raw.info['bads'] = ['O2', 'Iz']

    if (np.logical_and(bids.substring == 'sub-02', np.logical_and(day_count == 1, settings.testtrain == 1))):
        raw.info['bads'] = ['Oz']
    if (np.logical_and(bids.substring == 'sub-10', np.logical_and(day_count == 1, settings.testtrain == 1))):
        raw.info['bads'] = ['Oz']
    # sub 52 day 4- particularly noisy everywhere...

    # plt.show()
    # tmp = input('check the eeg data')

    eeg_data = raw.copy().pick_types(eeg=True, exclude=['TRIG'])
    eeg_data.info.set_montage(montageuse)
    eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=False)

    # Filter Data
    eeg_data_interp.filter(l_freq=1, h_freq=45, h_trans_bandwidth=0.1)

    #plot results
    eeg_data_interp.plot(remove_dc=False, scalings=dict(eeg=50))

    return raw, events, eeg_data_interp

def within_subjects_error(x):
    ''' calculate within subjects error for x
    arguments: x - assumes subjects as rows and conditions as columns
    returns: error - within subjects error for each condition
    '''
    rows, cols = x.shape
    subject_mean = np.nanmean(x, axis=1)
    grand_mean = np.nanmean(subject_mean, axis=0)

    x_2 = x - (np.tile(subject_mean, [ cols,1]).T - grand_mean)
    error = np.nanstd(x_2, axis=0) / np.sqrt(rows)
    return error
