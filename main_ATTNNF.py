# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import matplotlib.pyplot as plt

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

    # Directories
    direct_dataroot = Path("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Data/")
    direct_resultsroot = Path("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/")

    # relevant triggers
    trig_cuestart_cuediff = np.array(([121, 122], [123, 124]))

    # initialise
    def __init__(self, attntrained):
        self.attntrained = attntrained

        # get correct subject indices
        if (self.attntrained == 0):
            self.subsIDX = np.array(([1]))
        else:
            self.subsIDX = np.array(([10]))
        self.num_subs = len(self.subsIDX)


    def get_settings_EEG_prepost(self):
        # Task Settings
        self.testtrain = 0 # 0 = test, 1 = train
        self.task = 0 # 0 = motion descrim, 1 = visual search, 2 = n-back
        self.num_days = 2
        # self.daysuse = [1, 4]
        self.daysuse = [1]

        # EEG settings
        self.samplingfreq = 1200
        self.num_electrodes = 9

        # Timing Settings
        self.timelimits = np.array([0, 6]) # Epoch start and end time (seconds) relative to cue onset
        self.zeropading = 2
        self.timelimits_zeropad = np.array([self.timelimits[0]-self.zeropading, self.timelimits[1]-self.zeropading])
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
    timelimits_data[0] = timelimits_data[0] + 1

    num_seconds = timelimits[1] - timelimits[0]
    num_datapoints =  timelimits_data[1] -  timelimits_data[0]  + 1

    timepoints = np.arange(timelimits[0], timelimits[1] - 1 / samplingfreq, 1 / samplingfreq) # Time (in seconds) of each sample in epoch relative to trigger
    frequencypoints = np.arange(0, samplingfreq - 1 / num_seconds, 1 / num_seconds) # frequency (in Hz) of each sample in epoch after fourier transform

    zeropoint = np.argmin(np.abs(timepoints - 0))
    return timelimits_data, timepoints, frequencypoints, zeropoint



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
    timelimits_data, timepoints, frequencypoints, zeropoint = get_timing_variables(settings.timelimits,settings.samplingfreq)
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = get_timing_variables(settings.timelimits_zeropad,settings.samplingfreq)

    # iterate through test days
    for day_count, day_val in enumerate(settings.daysuse):
        # get file names
        bids = BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # decide which EEG file to use
        possiblefiles = []
        filesizes = []
        for filesfound in bids.direct_data_eeg.glob(bids.filename_eeg + "*.vhdr"):
            filesizes.append(filesfound.stat().st_size)
            possiblefiles.append(filesfound)

        file2useIDX = np.argmax(filesizes) # get the biggest file (there are often smaller shorter accidental recordings)
        file2use = possiblefiles[file2useIDX]

        # load EEG file
        raw = mne.io.read_raw_brainvision(file2use, preload=True)
        eeg_data = raw.copy()
        eeg_data.drop_channels('TRIG') # don't include the trigger channel in the main EEG data
        raw.pick_channels(['TRIG']) # do save the trigger channel though

        # print relevant info
        print(eeg_data.info)
        print(raw.info)

        # get triggers
        triggerchannel = raw['TRIG', :]
        # plt.plot(triggerchannel[1], triggerchannel[0].T) # plot

        triggerchannel_squeeze = np.squeeze(triggerchannel[0])
        tmp = np.hstack([0, np.diff(triggerchannel_squeeze)])
        trig_latency = np.squeeze(np.where(tmp>0)) # find latencies of changes
        trig_val = triggerchannel_squeeze[trig_latency]*1e6 # get the values of triggers at these points (multiply because the loading shrinks the data values for some reason)

        plt.stem(trig_latency, trig_val) # plot

        # Filter Data
        # eeg_data_filt = mne.filter.notch_filter(eeg_data, settings.samplingfreq, freqs=50, notch_widths=1)

        dat2filt = eeg_data[np.arange(9), :]
        dat2filt2 = dat2filt[0]*1e6 # multiply because the loading shrinks the data values for some reason
        eeg_filt = mne.filter.filter_data(dat2filt2, settings.samplingfreq, l_freq = 1, h_freq = 45, h_trans_bandwidth =0.1)

        # h = mne.filter.create_filter(dat2filt2, settings.samplingfreq, l_freq = 1, h_freq = 45, h_trans_bandwidth =0.1)
        # mne.viz.plot_filter(h,  settings.samplingfreq)

        plt.plot(dat2filt[1], dat2filt2[6,:].T)
        plt.plot(dat2filt[1], eeg_filt[6,:].T)

        # Epoch for the four attention conditions
        for cuetype in np.arange(2):
            for level in np.arange(2):
                condition_trigs = settings.trig_cuestart_cuediff[cuetype][level]

                idx = np.where(trig_val, condition_trigs)
                condition_latency = trig_latency[idx]

