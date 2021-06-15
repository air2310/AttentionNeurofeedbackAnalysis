import numpy as np
from pathlib import Path

# setup Classes and functions
class SetupMetaData:
    # Describe the options for settings
    string_attntrained = ["Space", "Feature", "Sham"]
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

    darkteal_ = "#122F41"
    medteal_ = "#058586"
    lightteal_ = "#4EB99F"
    yellow_ = "#F2B035"
    orange_ = "#EC553A"
    red_ = "#C11F3A"


    # initialise
    def __init__(self, attntrained):
        self.attntrained = attntrained

        # get correct subject indices
        if (self.attntrained == 0): # Space
            self.subsIDX =        np.array(([10, 11, 19, 22, 28, 29, 38, 43, 45, 46, 49, 52, 53, 54, 59, 60, 64, 71, 74, 79, 81, 84, 85, 90, 94, 97, 99, 104, 107, 112,      118, 123, 125, 128]))
            self.subsIDXcollate = np.array(([10, 11, 19, 22, 28, 29, 38, 43, 45, 46, 49, 52, 53, 54, 59, 60, 64, 71, 74, 79, 81, 84, 85, 90, 94, 97, 99, 104, 107, 112,      118, 123, 125, 128 ])) #, 19, 22, 28, 29, 43, 45, 46, 49, 52, 53, 54, 59, 60]))
            self.subsIDXall =     np.array(([10, 11, 19, 22, 28, 29, 38, 43, 45, 46, 49, 52, 53, 54, 59, 60, 64, 71, 74, 79, 81, 84, 85, 90, 94, 97, 99, 104, 107, 112, 113, 118, 123, 125, 128]))
            # 35

        if (self.attntrained == 1):  # Feature
            self.subsIDX =        np.array(([1, 2, 4, 8, 9, 18, 23, 41, 47, 57, 58, 63, 66, 67, 68, 69, 70, 72, 73, 76, 77, 78, 80, 86, 87, 89, 92, 100, 101, 102, 106, 110, 116, 117, 119, 120, 121]))
            self.subsIDXcollate = np.array(([1, 2, 4, 8, 9, 18, 23, 41, 47, 57, 58, 63, 66, 67, 68, 69, 70, 72, 73, 76, 77, 78, 80, 86, 87, 89, 92, 100, 101, 102, 106, 110, 116, 117, 119, 120, 121])) #np.array(([1, 2, 4, 8, 9, 18, 21, 23, 41, 47, 57, 58,63, 66, 67,68, 69 ]))
            self.subsIDXall =     np.array(([1, 2, 4, 8, 9, 18, 23, 41, 47, 57, 58, 63, 66, 67, 68, 69, 70, 72, 73, 76, 77, 78, 80, 86, 87, 89, 92, 100, 101, 102, 106, 110, 116, 117, 119, 120, 121]))
            # 37

        if (self.attntrained == 2):  # Sham
            self.subsIDX =        np.array(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39]))
            self.subsIDXcollate = np.array(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39]))
            self.subsIDXall =     np.array(([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 37, 39]))
            # 37

        self.num_subs = len(self.subsIDXcollate)


    def get_settings_EEG_prepost(self): # settings specific to pre vs post training EEG analysis
        # Task Settings
        self.testtrain = 0 # 0 = test, 1 = train
        self.task = 0 # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        self.daysuse = [1, 4]
        self.num_trials = 192
        self.num_conditions = 4
        self.num_conditionsRSA = 32
        # EEG settings
        self.samplingfreq = 1200
        self.num_electrodes = 9

        # Timing Settings
        self.timelimits = np.array([0, 6]) # Epoch start and end time (seconds) relative to cue onset
        self.zeropading = 2
        self.timelimits_zeropad = np.array([self.timelimits[0]-self.zeropading, self.timelimits[1]+self.zeropading])

        self.timelimits_motionepochs = np.array([-1, 2]) # Epoch start and end time (seconds) relative to cue onset
        self.timelimits_RSA = np.array([-0.5, 1.5])  # Epoch start and end time (seconds) relative to cue onset

        # relevant triggers
        trig_cuestart_cuediff = {'Feat/Black': 121, 'Feat/White': 122,
                                 'Space/Left_diag': 123,
                                 'Space/Right_diag': 124}  # will be different triggers for training days

        # Motion Onset triggers - programmed version and fixed version.
        self.num_relativepositions = 2 # top or bottom patch
        self.trig_motiononset = np.zeros((4, self.num_spaces, self.num_features, self.num_relativepositions)) # [FASA, FUSA, FASU, FUFU]

        self.trig_motiononset[: ,0, 0, 0] = [1, 2, 3, 4]
        self.trig_motiononset[:, 1, 0, 0] = self.trig_motiononset[:, 0, 0, 0] + 4

        self.trig_motiononset[:, :, 1, 0] = self.trig_motiononset[:, :, 0, 0] + 8
        self.trig_motiononset[:, :, 0, 1] = self.trig_motiononset[:, :, 0, 0] + 8 # This was a mistake in the origional code that we'll fix using the ground truth cue trigs
        self.trig_motiononset[:, :, 1, 1] = self.trig_motiononset[:, :, 0, 1] + 8

        self.trig_motiononset_new = np.zeros((self.num_attnstates, self.num_levels, self.num_attd_unattd)) # cue (space, feat), where was cued (level a, level b), where was movement (cued, uncued)
        self.trig_motiononset_new[:, 0, 0] = [1, 2]
        self.trig_motiononset_new[:, 1, 0] = [3, 4]
        self.trig_motiononset_new[:, 0, 1] = [5, 6]
        self.trig_motiononset_new[:, 1, 1] = [7, 8]


        return self

    def get_settings_behave_prepost(self):
        self.testtrain = 0  # 0 = test, 1 = train
        self.task = 0  # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        self.daysuse = [1, 4]
        self.num_trials = 192
        self.num_conditions = 4
        self.trialduration = 8
        self.cueduration = 2

        # response options
        self.responseopts_miss = 0
        self.responseopts_correct = 1
        self.responseopts_incorrect = 2

        self.responseopts_correctreject = 3
        self.responseopts_falsealarm = 4
        self.responseopts_falsealarm_incorrect = 5

        # timing settings
        self.mon_ref = 144
        self.responseperiod = np.round(np.array([0.3, 1.2]) * self.mon_ref) # max time = 500 ms motion epoch + min 400 ms between events + 300 ms period where it would be too early to respond to the next event.
        # self.responseperiod = np.round(np.array([0.3, 1.75]) * self.mon_ref)

        self.directions =  np.array([0, 90, 180, 270])

        self.trialattntype = ['Feature', 'Space', 'Both']
        self.string_attntype = ["Space", "Feature"]

        return self

    def get_settings_behave_duringNF(self):
        self.testtrain = 1  # 0 = test, 1 = train
        self.task = 0  # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 3
        self.daysuse = [1, 2, 3]
        self.num_trials = 256
        self.num_conditions = 4
        self.num_movements = 5
        self.trialduration = 8
        self.cueduration = 2

        # response options
        self.responseopts_miss = 0
        self.responseopts_correct = 1
        self.responseopts_incorrect = 2


        self.responseopts_correctreject = 3
        self.responseopts_falsealarm = 4
        self.responseopts_falsealarm_incorrect = 5

        # timing settings
        self.mon_ref = 144
        # self.responseperiod = np.round(np.array([0.3, 1.5]) * self.mon_ref)
        self.responseperiod = np.round(np.array([0.3, 1.75]) * self.mon_ref)

        self.directions =  np.array([0, 90, 180, 270])

        self.trialattntype = ['Feature', 'Space', 'Both']
        self.string_attntype = ["Space", "Feature"]

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


def get_eeg_data(bids, day_count, day_val, settings):
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
    if np.any(np.isin(np.array(['sub-68', 'sub-89', 'sub-119']), bids.substring)): #(bids.substring == 'sub-68'): Sometimes an especially short trigger is detected and we need to set a shorter min duration for triggers.
                events = mne.find_events(raw, stim_channel="TRIG", min_duration=4/raw.info['sfreq'])
    else:
        events = mne.find_events(raw, stim_channel="TRIG")

    mne.viz.plot_events(events, raw.info['sfreq'], raw.first_samp)
    print('Found %s events, first five:' % len(events))
    print(events[:5])

    # set bad channels to interpolate
    if settings.attntrained == 0:
        if np.logical_and(bids.substring == 'sub-53', day_val == 1):
            raw.info['bads'] = ['Iz']
        if bids.substring == 'sub-112': # check on this
            raw.info['bads'] = ['O1', 'O2']
        if np.logical_and(bids.substring == 'sub-10', day_val == 2):
            raw.info['bads'] = ['Oz']
        if np.logical_and(bids.substring == 'sub-125', day_val == 1):
            raw.info['bads'] = ['O1']
        if np.logical_and(bids.substring == 'sub-125', day_val == 2):
            raw.info['bads'] = ['PO4']
        if np.logical_and(bids.substring == 'sub-125', day_val == 3):
            raw.info['bads'] = ['PO4']

    if settings.attntrained == 1:
        if np.logical_and(bids.substring == 'sub-02', day_val == 1):
            raw.info['bads'] = ['O1']
        if np.logical_and(bids.substring == 'sub-02', day_val == 2):
            raw.info['bads'] = ['Oz']
        if np.logical_and(bids.substring == 'sub-09', day_val == 1):
            raw.info['bads'] = ['PO4']
        if np.logical_and(bids.substring == 'sub-23', day_val == 1):
            raw.info['bads'] = ['Iz']
        if np.logical_and(bids.substring == 'sub-47', day_val == 1):
            raw.info['bads'] = ['PO8']
        if np.logical_and(bids.substring == 'sub-70', day_val == 1):
            raw.info['bads'] = ['O2', 'Iz']
        if np.logical_and(bids.substring == 'sub-121', day_val == 1):
            raw.info['bads'] = ['Iz']


    if settings.attntrained == 2:
        if np.logical_and(bids.substring == 'sub-03', day_val == 4):
            raw.info['bads'] = ['PO3']
        if np.logical_and(bids.substring == 'sub-14', day_val == 4):
            raw.info['bads'] = ['Oz']
        if np.logical_and(bids.substring == 'sub-15', day_val == 4):
            raw.info['bads'] = ['O1']
        if np.logical_and(bids.substring == 'sub-16', day_val == 4):
            raw.info['bads'] = ['POz']
        if np.logical_and(bids.substring == 'sub-19', day_val == 1):
            raw.info['bads'] = ['POz']
        if np.logical_and(bids.substring == 'sub-20', day_val == 2):
            raw.info['bads'] = ['POz']
        if np.logical_and(bids.substring == 'sub-21', day_val == 1):
            raw.info['bads'] = ['Iz']
        if np.logical_and(bids.substring == 'sub-22', day_val == 4):
            raw.info['bads'] = ['Iz']
        if np.logical_and(bids.substring == 'sub-24', day_val == 1):
            raw.info['bads'] = ['POz', 'PO4']
        if np.logical_and(bids.substring == 'sub-28', day_val == 2):
            raw.info['bads'] = ['POz', 'PO4']
        if np.logical_and(bids.substring == 'sub-28', day_val == 3):
            raw.info['bads'] = ['POz', 'PO4']
        if np.logical_and(bids.substring == 'sub-29', day_val == 1):
            raw.info['bads'] = ['Oz']
        if np.logical_and(bids.substring == 'sub-30', day_val == 1):
            raw.info['bads'] = ['POz']
        if np.logical_and(bids.substring == 'sub-30', day_val == 4):
            raw.info['bads'] = ['PO4']
        if np.logical_and(bids.substring == 'sub-32', day_val == 2):
            raw.info['bads'] = ['POz']
        if np.logical_and(bids.substring == 'sub-34', day_val == 2):
            raw.info['bads'] = ['PO7']
        if np.logical_and(bids.substring == 'sub-37', day_val == 1):
            raw.info['bads'] = ['PO4']
        if np.logical_and(bids.substring == 'sub-37', day_val == 4):
            raw.info['bads'] = ['Iz', 'O2']


    # plt.show()
    # tmp = input('check the eeg data')

    eeg_data = raw.copy().pick_types(eeg=True, exclude=['TRIG'])
    eeg_data.info.set_montage(montageuse)
    eeg_data_interp = eeg_data.copy().interpolate_bads(reset_bads=True)

    # Filter Data
    eeg_data_interp.filter(l_freq=1, h_freq=45, h_trans_bandwidth=0.1)

    #plot results
    # eeg_data_interp.plot(remove_dc=False, scalings=dict(eeg=50))

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
