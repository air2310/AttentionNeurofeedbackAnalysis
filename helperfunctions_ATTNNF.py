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

    # colours
    darkteal = [18 / 255, 47 / 255, 65 / 255]
    medteal = [5 / 255, 133 / 255, 134 / 255]
    lightteal = [78 / 255, 185 / 255, 159 / 255]
    yellow = [242 / 255, 176 / 255, 53 / 255]
    orange = [236 / 255, 85 / 255, 58 / 255]

    # initialise
    def __init__(self, attntrained):
        self.attntrained = attntrained

        # get correct subject indices
        if (self.attntrained == 0): # Space
            self.subsIDX = np.array(([10, 11, 19, 22, 28, 29, 43, 45, 46, 49, 52, 53]))
            self.subsIDXcollate = np.array(([10, 11, 19, 22, 28, 29, 43, 45, 46, 49, 52, 53]))
            self.subsIDXall = np.array(([10, 11, 19, 22, 28, 29, 43, 45, 46, 49, 52, 53]))

        else: # Feature
            self.subsIDX = np.array(([  2, 4, 8, 9, 18, 21, 23, 41, 47])) # 1, 2,
            self.subsIDXcollate = np.array(([1, 2, 4, 8, 9, 18, 21, 23, 41, 47]))
            self.subsIDXall = np.array(([1, 2, 4, 8, 9, 18, 21, 23, 41, 47 ]))
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

        self.direct_results_group = settings.direct_resultsroot / Path(
            'Train' + settings.string_attntrained[settings.attntrained] + '/group/')
        self.direct_results_group.mkdir(parents=True, exist_ok=True)

        # filenames
        self.filename_eeg = self.casestring + '_eeg'
        self.filename_chan = self.casestring + '_channels'
        self.filename_evt = self.casestring + '_events'
        self.filename_behave = self.casestring + '_behav'
        self.filename_vissearch = self.casestring + '_vissearch'
        self.filename_nback = self.casestring + 'nback'

def get_timing_variables(timelimits,samplingfreq):
    timelimits_data = timelimits * samplingfreq

    num_seconds = timelimits[1] - timelimits[0]
    num_datapoints =  timelimits_data[1] -  timelimits_data[0]

    timepoints = np.arange(timelimits[0], timelimits[1] , 1 / samplingfreq) # Time (in seconds) of each sample in epoch relative to trigger
    frequencypoints = np.arange(0, samplingfreq , 1 / num_seconds) # frequency (in Hz) of each sample in epoch after fourier transform

    zeropoint = np.argmin(np.abs(timepoints - 0))
    return timelimits_data, timepoints, frequencypoints, zeropoint

def within_subjects_error(x):
    ''' calculate within subjects error for x
    arguments: x - assumes subjects as rows and conditions as columns
    returns: error - within subjects error for each condition
    '''
    rows, cols = x.shape
    subject_mean =


    return error
    # function[ebars] = ws_bars(x)
    #
    # % within    subjects    error    bars    % assumessubject as row and condition as column
    #
    # [rows, cols] = size(x);
    #
    # sub_mean = nanmean(x, 2); % disp(sub_mean);
    # grand_mean = nanmean(sub_mean, 1);
    #
    # x = x - (repmat(sub_mean, 1, cols) - grand_mean);
    #
    # ebars = nanstd(x) / sqrt(rows); % nanstd



