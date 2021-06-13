import numpy as np
from pathlib import Path
import mne
import helperfunctions_ATTNNF as helper
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def getMotionEventInfo(settings, sub_val, day_val):
    import h5py
    # get task specific settings
    settings = settings.get_settings_behave_prepost()

    # get file names
    bids = helper.BIDS_FileNaming(sub_val, settings, day_val)

    # decide which file to use
    possiblefiles = []
    filesizes = []
    for filesfound in bids.direct_data_behave.glob(bids.filename_behave + "*.mat"):
        filesizes.append(filesfound.stat().st_size)
        possiblefiles.append(filesfound)

    if any(filesizes):
        file2useIDX = np.argmax(filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
        file2use = possiblefiles[file2useIDX]

    # load data
    F = h5py.File(file2use, 'r')

    ### get variables of interest
    # Response Data
    MovePos = np.array(F['DATA']['MovePosAbs__Moveorder'])
    MoveFeat = np.array(F['DATA']['FeatMove__Moveorder'])

    return MovePos, MoveFeat

def participantRSA(settings, sub_val):
    print('analysing SSVEP amplitudes pre Vs. post training')

    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # get timing settings

    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_RSA, settings.samplingfreq)

    # preallocate
    ERPs_days = np.empty(( settings.num_electrodes, len(timepoints_zp) + 1, settings.num_conditionsRSA, settings.num_days))
    ERPs_days[:] = np.nan

    # iterate through test days to get data
    for day_count, day_val in enumerate(settings.daysuse):

        print(day_val)
        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)
        print(bids.casestring)

        # get EEG data
        raw, events, eeg_data_interp = helper.get_eeg_data(bids, day_count, day_val, settings)

        ############## fix FASA/FUSA trigger system for pre-post trials (we implemented a trigger system based on the way triggers were presented during NF. described in detail in the program script).
        idx_feat = np.where(np.logical_or(events[:, 2] == 121, events[:, 2] == 122)) # Get trial triggers
        idx_space = np.where(np.logical_or(events[:, 2] == 123, events[:, 2] == 124))

        tmp = np.zeros((len(events))) # Decide which trials are space and feature cues
        tmp[idx_space], tmp[idx_feat] = 1, 2
        cues = np.where(tmp>0)[0]

        MovePos, MoveFeat = getMotionEventInfo(settings, sub_val, day_val) # Get behavioural saved data on what moved where and when

        # Set up a new trigger scheme
        settings.num_positions = 4
        movetypetrigs = np.array(([[1, 2, 3, 4], [5, 6, 7, 8]]))
        newtrigs = np.empty((settings.num_features, settings.num_positions, settings.num_levels, settings.num_attnstates))
        newtrigs[:, :, 0, 0] = movetypetrigs
        newtrigs[:, :, 1, 0] = movetypetrigs + 8
        newtrigs[:, :, 0, 1] = movetypetrigs + 8 + 8
        newtrigs[:, :, 1, 1] = movetypetrigs + 8 + 8 + 8

        # Allocate these new triggers!
        for i in np.arange(len(cues)):
            # get events to edit
            if i == len(cues)-1:
                tmpevents = events[cues[i] + 1: -1, :]
            else:
                tmpevents = events[cues[i]+1 : cues[i+1]-1,:]

            # generate new triggers
            motiontrigs = np.where(np.isin(tmpevents[:,2], settings.trig_motiononset[:]))[0]

            for j in np.arange(len(motiontrigs)):
                cuetype = int(tmp[cues[i]]-1) # 0 = space, 1 = feat
                cuelevel = abs((events[cues[i], 2]-120) % 2 -1) # 0 = level 1, 1 = level 2
                MovePosuse = int(MovePos[j,i] - 1)
                MoveFeatuse = int(MoveFeat[j, i] -1)

                newtrig = newtrigs[MoveFeatuse, MovePosuse, cuelevel, cuetype]
                tmpevents[motiontrigs[j],2] = newtrig

            # replace edited triggers
            if i == len(cues) - 1:
                events[cues[i] + 1: -1, :] = tmpevents
            else:
                events[cues[i] + 1: cues[i + 1] - 1, :] = tmpevents

        # Epoch to events of interest
        # np.empty((settings.num_features, settings.num_positions, settings.num_levels, settings.num_attnstates))
        event_id = {'cue_S\_move_B1': 1, 'cue_S\_move_B2': 2, 'cue_S\_move_B3': 3, 'cue_S\_move_B4': 4,
                    'cue_S\_move_W1': 5, 'cue_S\_move_W2': 6, 'cue_S\_move_W3': 7, 'cue_S\_move_W4': 8,
                    'cue_S/_move_B1': 9, 'cue_S/_move_B2': 10, 'cue_S/_move_B3': 11, 'cue_S/_move_B4': 12,
                    'cue_S/_move_W1': 13, 'cue_S/_move_W2': 14, 'cue_S/_move_W3': 15, 'cue_S/_move_W4': 16,
                    'cue_F\_move_B1': 17, 'cue_F\_move_B2': 18, 'cue_F\_move_B3': 19, 'cue_F\_move_B4': 20,
                    'cue_F\_move_W1': 21, 'cue_F\_move_W2': 22, 'cue_F\_move_W3': 23, 'cue_F\_move_W4': 24,
                    'cue_F/_move_B1': 25, 'cue_F/_move_B2': 26, 'cue_F/_move_B3': 27, 'cue_F/_move_B4': 28,
                    'cue_F/_move_W1': 29, 'cue_F/_move_W2': 30, 'cue_F/_move_W3': 31, 'cue_F/_move_W4': 32
                    }  # will be different triggers for training days

        epochs = mne.Epochs(eeg_data_interp, events, event_id=event_id, tmin=settings.timelimits_RSA[0],
                            tmax=settings.timelimits_RSA[1],
                            baseline=(0, 1 / settings.samplingfreq), picks=np.arange(settings.num_electrodes),
                            reject=dict(eeg=400), detrend=1)  #

        epochs.drop_bad()

        for ii, label in enumerate(event_id.keys()):
            ERPs_days[:, :, ii, day_count] = np.mean(epochs[label].get_data(), axis=0)


    # Get FFT
    timepoints_zp = epochs.times
    from scipy.fft import fft, fftfreq, fftshift
    import matplotlib.pyplot as plt
    fftdat = np.abs(fft(ERPs_days, axis=1)) / len(epochs.times)
    freq = fftfreq(len(epochs.times), d=1 / settings.samplingfreq)  # get frequency bins


    # Plot FFT
    # fig, (ax1, ax2) = plt.subplots(2,1, figsize=(10, 5))
    # chanmeanfft = np.mean(fftdat, axis=0)
    # for day_count in np.arange(2):
    #     if (day_count == 0): axuse = ax1
    #     if (day_count == 1): axuse = ax2
    #     for i in np.arange(32):
    #         axuse.plot(freq, chanmeanfft[:, i, day_count].T, '-')
    #     axuse.set_xlim(2, 20)
    #     axuse.set_ylim(0, 1)
    #     axuse.set_title(settings.string_prepost[day_count])
    # titlestring = bids.substring + 'FFT Spectrum RSA'
    # fig.suptitle(titlestring)


    # get indices for frequencies of interest
    hz_attn_index = list()
    for space_count, space in enumerate(['Left_diag', 'Right_diag']):
        for feat_count, feat in enumerate(['Black', 'White']):
            hz_attn_index.append(np.argmin(np.abs(freq - settings.hz_attn[space_count, feat_count]))) # ['\B \W'], ['/B /W']

    settings.num_Hz = 4
    SSVEPS_RSA = fftdat[:,hz_attn_index, :, :].reshape(settings.num_electrodes*settings.num_Hz, settings.num_conditionsRSA, settings.num_days)


    # Run RSA!
    import scipy.stats as stats

    RDM = np.zeros((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days))
    RDM[:] = np.nan

    for day_count, day_val in enumerate(settings.daysuse):
        for conditionA in np.arange(settings.num_conditionsRSA):
            for conditionB in np.arange(settings.num_conditionsRSA):

                corr = stats.pearsonr(SSVEPS_RSA[:,conditionA, day_count] , SSVEPS_RSA[:,conditionB, day_count])
                correlation_distance = 1-corr[0]
                RDM[conditionA, conditionB, day_count] = correlation_distance


    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10, 5))
    ax1.imshow(RDM[:, :, 0], clim=(0.0, 2))
    ax1.set_title('Pre-Training')

    ax2.imshow(RDM[:, :, 1], clim=(0.0, 2))
    ax2.set_title('Post-Training')

    plt.set_cmap('jet')
    titlestring = bids.substring + 'RSA'
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    np.savez(bids.direct_results / Path(bids.substring + "RSA_data"), RDM=RDM, SSVEPS_RSA=SSVEPS_RSA)

def collate_RSA(settings):

    import scipy.stats as stats
    settings = helper.SetupMetaData(0)
    settings = settings.get_settings_EEG_prepost()
    settings.num_traininggroups = 3

    # list objects for individual participant RSA
    TrainingGroup = list()
    SUB_ID = list()
    SUB_ID_unique = list()
    RDM_Score = list()



    # Preallocate for group
    SSVEPS_RSA_Group = np.zeros((36, settings.num_conditionsRSA, settings.num_days, settings.num_traininggroups))
    RDM_Group = np.zeros((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days, settings.num_traininggroups))
    RDM_Group[:] = np.nan
    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_EEG_prepost()

        # preallocate for group
        settings.num_Hz = 4
        RDM_individual = np.zeros((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days, settings.num_subs))
        SSVEPS_RSA = np.zeros((settings.num_electrodes * settings.num_Hz, settings.num_conditionsRSA, settings.num_days, settings.num_subs))
        dissimilarity_days = np.zeros((settings.num_subs))
        SSVEPS_RSA[:] = np.nan
        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "RSA_data.npz"), allow_pickle=True) #, RDM=RDM, SSVEPS_RSA=SSVEPS_RSA)
            RDM_individual[:, :, :, sub_count] = results['RDM']
            SSVEPS_RSA[:, :, :, sub_count] = results['SSVEPS_RSA']



            # subtract mean value from each electrode and frequency so we're just targetting the variance across conditions.
            SSVEPS_RSA[:, :, 0, sub_count] = SSVEPS_RSA[:, :, 0, sub_count] - SSVEPS_RSA[:, :, 0, sub_count].mean(axis=1)[:, None]
            SSVEPS_RSA[:, :, 1, sub_count] = SSVEPS_RSA[:, :, 1, sub_count] - SSVEPS_RSA[:, :, 1, sub_count].mean(axis=1)[:, None]

            # Run RSA on group data
            for day_count, day_val in enumerate(settings.daysuse):
                for conditionA in np.arange(settings.num_conditionsRSA):
                    for conditionB in np.arange(settings.num_conditionsRSA):
                        corr = stats.pearsonr(SSVEPS_RSA[:, conditionA, day_count, sub_count], SSVEPS_RSA[:, conditionB, day_count, sub_count])
                        correlation_distance = 1 - corr[0]
                        RDM_individual[conditionA, conditionB, day_count, sub_count] = correlation_distance



            # work out Representational dissimilarity across days
            day1 = np.array((np.nan))
            day4 = np.array((np.nan))
            for i in np.arange(settings.num_conditionsRSA):
                day1 = np.hstack((day1, RDM_individual[i, i + 1:, 0, sub_count]))
                day4 = np.hstack((day4, RDM_individual[i, i + 1:, 1, sub_count]))

            corr = stats.spearmanr(day1[1:], day4[1:])
            dissimilarity_days[sub_count] = 1 - corr[0]

            # Data frame stuff
            TrainingGroup.append(attntrained)
            SUB_ID.append(sub_val)
            SUB_ID_unique.append(sub_count + attntrainedcount*37)
            RDM_Score.append(dissimilarity_days[sub_count])


        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(RDM_individual[:, :, 0, :].mean(axis=2), clim=(0.0, 2))
        ax1.set_title('Pre-Training')

        ax2.imshow(RDM_individual[:, :, 1, :].mean(axis=2), clim=(0.0, 2))
        ax2.set_title('Post-Training')

        plt.set_cmap('jet')
        titlestring = attntrained + 'subject mean RSA'
        fig.suptitle(titlestring)
        # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


        # Collate data
        SSVEPS_RSA_Group[:, :, :, attntrainedcount] = SSVEPS_RSA.mean(axis=3)

        # subtract mean value from each electrode and frequency so we're just targetting the variance across conditions.
        SSVEPS_RSA_Group[:, :, 0, attntrainedcount] = SSVEPS_RSA_Group[:, :, 0, attntrainedcount] - SSVEPS_RSA_Group[:, :, 0, attntrainedcount].mean(axis=1)[:,None]
        SSVEPS_RSA_Group[:, :, 1, attntrainedcount] = SSVEPS_RSA_Group[:, :, 1, attntrainedcount] - SSVEPS_RSA_Group[:, :, 1, attntrainedcount].mean(axis=1)[:, None]

        # Run RSA on group data
        for day_count, day_val in enumerate(settings.daysuse):
            for conditionA in np.arange(settings.num_conditionsRSA):
                for conditionB in np.arange(settings.num_conditionsRSA):
                    corr = stats.spearmanr(SSVEPS_RSA_Group[:, conditionA, day_count, attntrainedcount], SSVEPS_RSA_Group[:, conditionB, day_count, attntrainedcount])
                    correlation_distance = 1 - corr[0]
                    RDM_Group[conditionA, conditionB, day_count, attntrainedcount] = correlation_distance

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        ax1.imshow(RDM_Group[:, :, 0, attntrainedcount], clim=(0.0, 2))
        ax1.set_title('Pre-Training')

        ax2.imshow(RDM_Group[:, :, 1, attntrainedcount], clim=(0.0, 2))
        ax2.set_title('Post-Training')

        plt.set_cmap('jet')
        titlestring = attntrained + ' RSA'
        fig.suptitle(titlestring)
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

        # work out Representational dissimilarity across days

        day1 = np.array((np.nan))
        day4 = np.array((np.nan))
        for i in np.arange(settings.num_conditionsRSA):
            day1 = np.hstack((day1, RDM_Group[i, i + 1:, 0, attntrainedcount]))
            day4 = np.hstack((day4, RDM_Group[i, i + 1:, 1, attntrainedcount]))

        corr = stats.spearmanr(day1[1:], day4[1:])
        print(1 - corr[0])



    # create dataframe of individual participant results
    data = {'SubID': SUB_ID, 'SUB_ID_unique': SUB_ID_unique, 'TrainingGroup': TrainingGroup,
            'RDM_Score': RDM_Score}
    df_RDM = pd.DataFrame(data)

    # stats n stuff.
    df_RDM.groupby('TrainingGroup').mean()
    stats.ttest_ind(df_RDM.loc[df_RDM.TrainingGroup.isin(["Feature"]), 'RDM_Score'], df_RDM.loc[df_RDM.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    stats.ttest_ind(df_RDM.loc[df_RDM.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM.loc[df_RDM.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    stats.ttest_ind(df_RDM.loc[df_RDM.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM.loc[df_RDM.TrainingGroup.isin(["Feature"]), 'RDM_Score'])

