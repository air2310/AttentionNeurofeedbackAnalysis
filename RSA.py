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
    from scipy.fft import fft, fftfreq, fftshift
    import matplotlib.pyplot as plt
    print('analysing SSVEP amplitudes pre Vs. post training')

    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # get timing settings

    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_RSA, settings.samplingfreq)

    # preallocate
    ERPs_days = np.empty(( settings.num_electrodes, len(timepoints_zp) + 1, settings.num_conditionsRSA, settings.num_days))
    ERPs_days[:] = np.nan

    FFTs_days = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_conditionsRSA, settings.num_days))
    FFTs_days[:] = np.nan

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

            # zeropad
            epochsuse = epochs[label].get_data()
            epochsuse[:, :, epochs.times < 0] = 0
            epochsuse[:, :, epochs.times > 1] = 0

            # Get FFT
            fftdat = np.abs(fft(epochsuse, axis=2)) / len(epochs.times)
            FFTs_days[:, :, ii, day_count] = np.mean(fftdat, axis=0)


    # zeropad
    ERPs_days_zp = ERPs_days
    ERPs_days_zp[:, epochs.times < 0, :, :] = 0
    ERPs_days_zp[:, epochs.times > 1, :, :] = 0

    # Get FFT
    fftdat = np.abs(fft(ERPs_days_zp, axis=1)) / len(epochs.times)
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

    # get single trial amp for Alpha
    frequse = freq[np.logical_and(freq < 11.9, freq > 8)]
    frequse = frequse[np.logical_or(frequse < 8.9, frequse > 9.1)]
    lowalpha = np.isin(freq, frequse[0:3])
    highalpha = np.isin(freq, frequse[3:])

    lowalphadat = FFTs_days[:, lowalpha, :, :].mean(axis=1)
    highalphadat = FFTs_days[:, highalpha, :, :].mean(axis=1)

    # get indices for frequencies of interest
    hz_attn_index = list()
    for space_count, space in enumerate(['Left_diag', 'Right_diag']):
        for feat_count, feat in enumerate(['Black', 'White']):
            hz_attn_index.append(np.argmin(np.abs(freq - settings.hz_attn[space_count, feat_count]))) # ['\B \W'], ['/B /W']

    settings.num_Hz = 4
    freqtagdat = fftdat[:,hz_attn_index, :, :].reshape(settings.num_electrodes*settings.num_Hz, settings.num_conditionsRSA, settings.num_days)

    SSVEPS_RSA = np.concatenate((freqtagdat, lowalphadat, highalphadat), axis=0)
    SSVEPS_RSA_plot = SSVEPS_RSA - SSVEPS_RSA.mean(axis=1)[:, None, :]

    # Run RSA!
    import scipy.stats as stats

    RDM = np.zeros((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days))
    RDM[:] = np.nan

    for day_count, day_val in enumerate(settings.daysuse):
        for conditionA in np.arange(settings.num_conditionsRSA):
            for conditionB in np.arange(settings.num_conditionsRSA):

                corr = stats.pearsonr(SSVEPS_RSA_plot[:,conditionA, day_count], SSVEPS_RSA_plot[:,conditionB, day_count])
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

    # list objects for bootstrapped group RSA
    TrainingGroup_grp = list()
    Bootstrap_grp = list()
    RDM_Score_grp = list()


    # Preallocate for group
    SSVEPS_RSA_Group = np.zeros((54, settings.num_conditionsRSA, settings.num_days, settings.num_traininggroups))
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
        SSVEPS_RSA = np.zeros((settings.num_electrodes * settings.num_Hz + settings.num_electrodes *2, settings.num_conditionsRSA, settings.num_days, settings.num_subs))
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
            SUB_ID.append( attntrained + str(sub_val))
            SUB_ID_unique.append(sub_count + attntrainedcount*37)
            RDM_Score.append(dissimilarity_days[sub_count])


        # Collate data
        # subtract mean value from each electrode and frequency so we're just targetting the variance across conditions.
        SSVEPS_RSA[:, :, 0, :] = np.divide(SSVEPS_RSA[:, :, 0, :] - SSVEPS_RSA[:, :, 0, :].mean(axis=1)[:, None, :], SSVEPS_RSA[:, :, 0, :].std(axis=1)[:, None, :])
        SSVEPS_RSA[:, :, 1, :] = np.divide(SSVEPS_RSA[:, :, 1, :] - SSVEPS_RSA[:, :, 1, :].mean(axis=1)[:, None, :], SSVEPS_RSA[:, :, 1, :].std(axis=1)[:, None, :])

        SSVEPS_RSA_Group[:, :, :, attntrainedcount] = SSVEPS_RSA.mean(axis=3)

        # # average across two types of alpha
        # valsuse = np.arange(settings.num_electrodes * settings.num_Hz,settings.num_electrodes * settings.num_Hz + settings.num_electrodes)
        # alphalow = SSVEPS_RSA_Group[valsuse, :, :, :]
        # valsuse = np.arange(settings.num_electrodes * settings.num_Hz + settings.num_electrodes, settings.num_electrodes * settings.num_Hz + settings.num_electrodes  + settings.num_electrodes)
        # alphahigh = SSVEPS_RSA_Group[valsuse, :, :, :]
        #
        # dat = np.mean([alphalow, alphahigh], axis=0)
        # SSVEPS_RSA_Groupuse = np.concatenate((SSVEPS_RSA_Group[np.arange(36), :, :, :], dat), axis=0)

        SSVEPS_RSA_Groupuse = SSVEPS_RSA_Group
        # Run RSA on group data
        for day_count, day_val in enumerate(settings.daysuse):
            for conditionA in np.arange(settings.num_conditionsRSA):
                for conditionB in np.arange(settings.num_conditionsRSA):
                    corr = stats.spearmanr(SSVEPS_RSA_Groupuse[:, conditionA, day_count, attntrainedcount], SSVEPS_RSA_Groupuse[:, conditionB, day_count, attntrainedcount])
                    correlation_distance = 1 - corr[0]
                    RDM_Group[conditionA, conditionB, day_count, attntrainedcount] = correlation_distance

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        im = ax1.imshow(RDM_Group[:, :, 0, attntrainedcount], clim=(0.0, 2))
        ax1.set_title('Pre-Training')

        im = ax2.imshow(RDM_Group[:, :, 1, attntrainedcount], clim=(0.0, 2))
        ax2.set_title('Post-Training')

        plt.set_cmap('jet')
        titlestring = attntrained + ' RSA'
        fig.suptitle(titlestring)
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.colorbar(im, ax=ax1)
        fig.colorbar(im, ax=ax2)
        titlestring = attntrained + ' RSAcolorbars'
        fig.suptitle(titlestring)
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

        day1 = np.array((np.nan))
        day4 = np.array((np.nan))
        for i in np.arange(settings.num_conditionsRSA):
            day1 = np.hstack((day1, RDM_Group[i, i + 1:, 0, attntrainedcount]))
            day4 = np.hstack((day4, RDM_Group[i, i + 1:, 1, attntrainedcount]))

        corr = stats.spearmanr(day1[1:], day4[1:])
        print(1-corr[0])

        ## Bootstrap analysis
        # Run RSA on group data
        num_bootstraps = 100
        RDM_Group_bootstrap = np.empty((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days, num_bootstraps))

        for bootstrap in np.arange(num_bootstraps):
            # work out Representational dissimilarity across days
            day1 = np.array((np.nan))
            day4 = np.array((np.nan))

            conduse = np.random.choice(settings.num_conditionsRSA, settings.num_conditionsRSA, replace=True)
            # conduse = np.arange(32)
            datuse = RDM_Group[:, conduse, :, attntrainedcount]

            for i in np.arange(settings.num_conditionsRSA):
                day1 = np.hstack((day1, datuse[i, i + 1:, 0]))
                day4 = np.hstack((day4, datuse[i, i + 1:, 1]))

            nans = np.logical_or(day1 ==0, day4==0)
            day1 = np.delete(day1, nans)
            day4 = np.delete(day4, nans)
            corr = stats.spearmanr(day1[1:], day4[1:])

            TrainingGroup_grp.append(attntrained)
            Bootstrap_grp.append(bootstrap)
            RDM_Score_grp.append(1 - corr[0])


    # create dataframe of group results
    data = {'TrainingGroup': TrainingGroup_grp, 'Bootstrap': Bootstrap_grp, 'RDM_Score': RDM_Score_grp}
    df_RDM_grp = pd.DataFrame(data)

    # stats n stuff.
    df_RDM_grp.groupby('TrainingGroup').mean()
    stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Feature"]), 'RDM_Score'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Feature"]), 'RDM_Score'])


    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange, settings.red]

    sns.swarmplot(x="TrainingGroup", y="RDM_Score", data=df_RDM_grp, color="0", alpha=0.3, ax=ax)
    sns.violinplot(x="TrainingGroup", y="RDM_Score", data=df_RDM_grp, palette=sns.color_palette(colors), style="ticks",
                   ax=ax, inner="box", alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Distance from pre-training')

    titlestring = 'RSA pre to post training distance'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')






    # # create dataframe of individual participant results
    # data = {'SubID': SUB_ID, 'SUB_ID_unique': SUB_ID_unique, 'TrainingGroup': TrainingGroup,
    #         'RDM_Score': RDM_Score}
    # df_RDM = pd.DataFrame(data)
    #
    # # stats n stuff.
    # df_RDM.groupby('TrainingGroup').mean()
    # stats.ttest_ind(df_RDM.loc[df_RDM.TrainingGroup.isin(["Feature"]), 'RDM_Score'], df_RDM.loc[df_RDM.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    # stats.ttest_ind(df_RDM.loc[df_RDM.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM.loc[df_RDM.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    # stats.ttest_ind(df_RDM.loc[df_RDM.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM.loc[df_RDM.TrainingGroup.isin(["Feature"]), 'RDM_Score'])
    #
    # ## Individual subject behaviour correlation
    # df_sensitivity, bids = load_MotionDiscrimBehaveResults(settings)
    #
    # # exclude data
    # df_RDM = df_RDM.loc[~df_sensitivity.exclude_spacetask,].copy()
    # df_sensitivity = df_sensitivity.loc[~df_sensitivity.exclude_spacetask,].copy()
    #
    # datuse = pd.concat([df_RDM, df_sensitivity[['sensitivity_train_spacecue', 'sensitivity_train_featcue', 'sensitivity_pre_spacecue', 'sensitivity_pre_featcue', 'sensitivity_post_spacecue', 'sensitivity_post_featcue']]], axis=1)
    #
    # # Correlate
    # corr = stats.pearsonr(df_RDM['RDM_Score'], df_sensitivity['sensitivity_train_spacecue'])
    # print(corr)
    #
    # # median split
    # highRDM = datuse.loc[datuse['RDM_Score'] > datuse['RDM_Score'].quantile(0.5), ]
    # lowRDM = datuse.loc[datuse['RDM_Score'] < datuse['RDM_Score'].quantile(0.5), ]
    #
    # highRDM['sensitivity_train_spacecue'].mean()
    # lowRDM['sensitivity_train_spacecue'].mean()
    #
    # stats.ttest_ind(highRDM['sensitivity_train_spacecue'], lowRDM['sensitivity_train_spacecue'])
    #
    #
    # # Plot
    # fig, axuse = plt.subplots(1, 1, figsize=(6, 6))
    # sns.scatterplot(data=datuse, x='RDM_Score', y='sensitivity_train_spacecue', ax=axuse, color=settings.darkteal_)
    # axuse.spines['top'].set_visible(False)
    # axuse.spines['right'].set_visible(False)


def load_MotionDiscrimBehaveResults(settings):

    # initialise
    sensitivity_pre_spacecue = list()
    sensitivity_post_spacecue = list()
    sensitivity_train_spacecue = list()
    sensitivity_pre_featcue = list()
    sensitivity_post_featcue = list()
    sensitivity_train_featcue = list()

    correct_pre_spacecue = list()
    correct_post_spacecue = list()
    correct_pre_featcue = list()
    correct_post_featcue = list()

    sub_vec = list()
    traingroup_vec = list()

    # get task specific settings
    settings = settings.get_settings_behave_prepost()

    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):  # cycle trough space, feature and sham train groups
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_behave_prepost()

        # file names
        bids = helper.BIDS_FileNaming(subject_idx=0, settings=settings, day_val=0)
        df_behaveresults_tmp = pd.read_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))

        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # Get Data for this participant
            datuse = df_behaveresults_tmp.loc[df_behaveresults_tmp.subIDval.isin([sub_val]), ].copy()

            day1 = datuse['Testday'].isin(['Day 1'])
            day4 = datuse['Testday'].isin(['Day 4'])
            space = datuse['Attention Type'].isin(['Space'])
            feat = datuse['Attention Type'].isin(['Feature'])

            sensitivity_pre_spacecue.append(datuse.loc[day1 & space, 'Sensitivity'].tolist()[0])
            sensitivity_post_spacecue.append(datuse.loc[day4 & space, 'Sensitivity'].tolist()[0])
            sensitivity_train_spacecue.append(datuse.loc[day4 & space, 'Sensitivity'].tolist()[0] - datuse.loc[day1 & space, 'Sensitivity'].tolist()[0])

            sensitivity_pre_featcue.append(datuse.loc[day1 & feat, 'Sensitivity'].tolist()[0])
            sensitivity_post_featcue.append(datuse.loc[day4 & feat, 'Sensitivity'].tolist()[0])
            sensitivity_train_featcue.append(datuse.loc[day4 & feat, 'Sensitivity'].tolist()[0] - datuse.loc[day1 & feat, 'Sensitivity'].tolist()[0])

            correct_pre_spacecue.append(datuse.loc[day1 & space, 'correct'].tolist()[0])
            correct_post_spacecue.append(datuse.loc[day4 & space, 'correct'].tolist()[0])
            correct_pre_featcue.append(datuse.loc[day1 & feat, 'correct'].tolist()[0])
            correct_post_featcue.append(datuse.loc[day4 & feat, 'correct'].tolist()[0])

            traingroup_vec.append(attntrained)
            sub_vec.append(attntrained + str(sub_val))

    data = {'SubID': sub_vec, 'TrainingGroup': traingroup_vec,
            'sensitivity_pre_spacecue': sensitivity_pre_spacecue, 'sensitivity_post_spacecue': sensitivity_post_spacecue, 'sensitivity_train_spacecue': sensitivity_train_spacecue,
            'sensitivity_pre_featcue': sensitivity_pre_featcue, 'sensitivity_post_featcue': sensitivity_post_featcue, 'sensitivity_train_featcue': sensitivity_train_featcue,
            'correct_pre_spacecue': correct_pre_spacecue, 'correct_post_spacecue': correct_post_spacecue, 'correct_pre_featcue': correct_pre_featcue, 'correct_post_featcue': correct_post_featcue}

    df_sensitivity = pd.DataFrame(data)

    # subjects to exlude
    exclude_sp = list()
    exclude_ft = list()
    cutoff = 10

    tmp = df_sensitivity['correct_pre_spacecue']
    exclude_sp.extend(df_sensitivity[tmp < cutoff]['SubID'].tolist())
    tmp = df_sensitivity['correct_post_spacecue']
    exclude_sp.extend(df_sensitivity[tmp < cutoff]['SubID'].tolist())
    tmp = df_sensitivity['correct_pre_featcue']
    exclude_ft.extend(df_sensitivity[tmp < cutoff]['SubID'].tolist())
    tmp = df_sensitivity['correct_post_featcue']
    exclude_ft.extend(df_sensitivity[tmp < cutoff]['SubID'].tolist())

    df_sensitivity['exclude_spacetask'] = False
    df_sensitivity['exclude_feattask'] = False

    df_sensitivity.loc[df_sensitivity['SubID'].isin(np.unique(exclude_sp)), 'exclude_spacetask'] = True
    df_sensitivity.loc[df_sensitivity['SubID'].isin(np.unique(exclude_ft)), 'exclude_feattask'] = True

    return df_sensitivity, bids


def collate_RSA_bybehave_old(settings):

    import scipy.stats as stats
    settings = helper.SetupMetaData(0)
    settings = settings.get_settings_EEG_prepost()
    settings.num_traininggroups = 3

    # get behave data and split participants by performance
    df_sensitivity, bids = load_MotionDiscrimBehaveResults(settings)

    # list objects for bootstrapped group RSA
    TrainingGroup_grp = list()
    Bootstrap_grp = list()
    RDM_Score_grp = list()
    BehaveTraining_grp = list()

    # Preallocate for group
    # SSVEPS_RSA_Group_high = np.zeros((54, settings.num_conditionsRSA, settings.num_days, settings.num_traininggroups))
    # SSVEPS_RSA_Group_low = np.zeros((54, settings.num_conditionsRSA, settings.num_days, settings.num_traininggroups))

    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_EEG_prepost()

        datuse = df_sensitivity.loc[df_sensitivity['TrainingGroup'].isin([attntrained]),]
        lowtrain = datuse.loc[datuse['sensitivity_pre_featcue'] < datuse['sensitivity_post_featcue'].quantile(.5), 'SubID']
        hightrain = datuse.loc[datuse['sensitivity_pre_featcue'] > datuse['sensitivity_post_featcue'].quantile(.5), 'SubID']

        ## preallocate for group
        settings.num_Hz = 4
        SSVEPS_RSA_low = np.zeros((settings.num_electrodes * settings.num_Hz + settings.num_electrodes *2, settings.num_conditionsRSA, settings.num_days, len(lowtrain)))
        SSVEPS_RSA_high = np.zeros((settings.num_electrodes * settings.num_Hz + settings.num_electrodes * 2, settings.num_conditionsRSA, settings.num_days, len(hightrain)))
        SSVEPS_RSA_low[:] = np.nan
        SSVEPS_RSA_high[:] = np.nan

        # iterate through subjects for individual subject analyses
        sub_count_low = -1
        sub_count_high = -1
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "RSA_data.npz"), allow_pickle=True) #, RDM=RDM, SSVEPS_RSA=SSVEPS_RSA)

            # sort by behaviour
            subcondstring = attntrained + str(sub_val)
            if any(hightrain.isin([subcondstring])):
                sub_count_high = sub_count_high+1
                SSVEPS_RSA_high[:, :, :, sub_count_high] = results['SSVEPS_RSA']
            if any(lowtrain.isin([subcondstring])):
                sub_count_low = sub_count_low+1
                SSVEPS_RSA_low[:, :, :, sub_count_low] = results['SSVEPS_RSA']

        # Collate data
        # subtract mean value from each electrode and frequency so we're just targetting the variance across conditions.
        SSVEPS_RSA_high[:, :, 0, :] = np.divide(SSVEPS_RSA_high[:, :, 0, :] - SSVEPS_RSA_high[:, :, 0, :].mean(axis=1)[:, None, :], SSVEPS_RSA_high[:, :, 0, :].std(axis=1)[:, None, :])
        SSVEPS_RSA_high[:, :, 1, :] = np.divide(SSVEPS_RSA_high[:, :, 1, :] - SSVEPS_RSA_high[:, :, 1, :].mean(axis=1)[:, None, :], SSVEPS_RSA_high[:, :, 1, :].std(axis=1)[:, None, :])

        SSVEPS_RSA_low[:, :, 0, :] = np.divide(SSVEPS_RSA_low[:, :, 0, :] - SSVEPS_RSA_low[:, :, 0, :].mean(axis=1)[:, None, :], SSVEPS_RSA_low[:, :, 0, :].std(axis=1)[:, None, :])
        SSVEPS_RSA_low[:, :, 1, :] = np.divide(SSVEPS_RSA_low[:, :, 1, :] - SSVEPS_RSA_low[:, :, 1, :].mean(axis=1)[:, None, :], SSVEPS_RSA_low[:, :, 1, :].std(axis=1)[:, None, :])

        SSVEPS_RSA_Group_high = SSVEPS_RSA_high.mean(axis=3)
        SSVEPS_RSA_Group_low = SSVEPS_RSA_low.mean(axis=3)

        for behavetrain, behavetrainstr in enumerate(['hightrain', 'lowtrain']):
            if behavetrain == 0:
                SSVEPS_RSA_Groupuse = SSVEPS_RSA_Group_high
            if behavetrain == 1:
                SSVEPS_RSA_Groupuse = SSVEPS_RSA_Group_low

            # Run RSA on group data
            RDM_Group = np.zeros((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days))
            RDM_Group[:] = np.nan
            for day_count, day_val in enumerate(settings.daysuse):
                for conditionA in np.arange(settings.num_conditionsRSA):
                    for conditionB in np.arange(settings.num_conditionsRSA):
                        corr = stats.spearmanr(SSVEPS_RSA_Groupuse[:, conditionA, day_count], SSVEPS_RSA_Groupuse[:, conditionB, day_count])
                        correlation_distance = 1 - corr[0]
                        RDM_Group[conditionA, conditionB, day_count] = correlation_distance

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            im = ax1.imshow(RDM_Group[:, :, 0], clim=(0.0, 2))
            ax1.set_title('Pre-Training')

            im = ax2.imshow(RDM_Group[:, :, 1], clim=(0.0, 2))
            ax2.set_title('Post-Training')

            plt.set_cmap('jet')
            titlestring = attntrained + ' RSA ' + behavetrainstr
            fig.suptitle(titlestring)
            plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
            plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

            # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            # fig.colorbar(im, ax=ax1)
            # fig.colorbar(im, ax=ax2)
            # titlestring = attntrained + ' RSAcolorbars'
            # fig.suptitle(titlestring)
            # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

            day1 = np.array((np.nan))
            day4 = np.array((np.nan))
            for i in np.arange(settings.num_conditionsRSA):
                day1 = np.hstack((day1, RDM_Group[i, i + 1:, 0]))
                day4 = np.hstack((day4, RDM_Group[i, i + 1:, 1]))

            corr = stats.spearmanr(day1[1:], day4[1:])
            print(1-corr[0])

            ## Bootstrap analysis
            # Run RSA on group data
            num_bootstraps = 100
            RDM_Group_bootstrap = np.empty((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days, num_bootstraps))

            for bootstrap in np.arange(num_bootstraps):
                # work out Representational dissimilarity across days
                day1 = np.array((np.nan))
                day4 = np.array((np.nan))

                conduse = np.random.choice(settings.num_conditionsRSA, settings.num_conditionsRSA, replace=True)
                # conduse = np.arange(32)
                datuse = RDM_Group[:, conduse, :]

                for i in np.arange(settings.num_conditionsRSA):
                    day1 = np.hstack((day1, datuse[i, i + 1:, 0]))
                    day4 = np.hstack((day4, datuse[i, i + 1:, 1]))

                nans = np.logical_or(day1 ==0, day4==0)
                day1 = np.delete(day1, nans)
                day4 = np.delete(day4, nans)
                corr = stats.spearmanr(day1[1:], day4[1:])

                BehaveTraining_grp.append(behavetrainstr)
                TrainingGroup_grp.append(attntrained)
                Bootstrap_grp.append(bootstrap)
                RDM_Score_grp.append(1 - corr[0])


    # create dataframe of group results
    data = {'TrainingGroup': TrainingGroup_grp, 'Bootstrap': Bootstrap_grp, 'RDM_Score': RDM_Score_grp, 'BehaveTraining_grp': BehaveTraining_grp}
    df_RDM_grp = pd.DataFrame(data)

    # stats n stuff.
    df_RDM_grp.groupby('TrainingGroup').mean()
    stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Feature"]), 'RDM_Score'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Sham"]), 'RDM_Score'])
    stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Space"]), 'RDM_Score'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Feature"]), 'RDM_Score'])


    fig, ax = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange, settings.red]

    sns.swarmplot(x="TrainingGroup", y="RDM_Score", hue = BehaveTraining_grp, data=df_RDM_grp, color="0", alpha=0.3, ax=ax, dodge=True)
    sns.violinplot(x="TrainingGroup", y="RDM_Score", hue = BehaveTraining_grp,data=df_RDM_grp, palette=sns.color_palette(colors), style="ticks",
                   ax=ax, inner="box", alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ax.set_ylabel('Distance from pre-training')

    titlestring = 'RSA pre to post training distance'
    plt.suptitle(titlestring)
    # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')


def collate_RSA_bybehave(settings):

    import scipy.stats as stats
    settings = helper.SetupMetaData(0)
    settings = settings.get_settings_EEG_prepost()
    settings.num_traininggroups = 3

    # get behave data and split participants by performance
    df_sensitivity, bids = load_MotionDiscrimBehaveResults(settings)

    for cuetype, cuetypestring in enumerate(["Space Task", "Feature Task"]):
        # get performance for each cuetype
        if cuetype == 0:
            df_sensitivity['perform'] = (df_sensitivity['sensitivity_pre_spacecue'] + df_sensitivity['sensitivity_post_spacecue']) / 2
        else:
            df_sensitivity['perform'] = (df_sensitivity['sensitivity_pre_featcue'] + df_sensitivity['sensitivity_post_featcue']) / 2

        # get top 25% of people from each group - even numbers of people then get removed from each group.
        datuse = df_sensitivity.loc[df_sensitivity['TrainingGroup'].isin(["Space"]),]
        highperformers = datuse.loc[datuse['perform'] > datuse['perform'].quantile(.80), 'SubID']
        datuse = df_sensitivity.loc[df_sensitivity['TrainingGroup'].isin(["Feature"]),]
        highperformers = pd.concat([highperformers, datuse.loc[datuse['perform'] > datuse['perform'].quantile(.80), 'SubID']], axis=0)
        datuse = df_sensitivity.loc[df_sensitivity['TrainingGroup'].isin(["Sham"]),]
        highperformers = pd.concat([highperformers, datuse.loc[datuse['perform'] > datuse['perform'].quantile(.80), 'SubID']], axis=0)

        ## preallocate for group
        settings.num_Hz = 4
        SSVEPS_RSA_high = np.zeros((settings.num_electrodes * settings.num_Hz + settings.num_electrodes * 2, settings.num_conditionsRSA, settings.num_days, len(highperformers)))
        SSVEPS_RSA_high[:] = np.nan

        # cycle trough space and feature train groups
        sub_count_high = -1
        for attntrainedcount, attntrained in enumerate(settings.string_attntrained):
            # setup generic settings
            settings = helper.SetupMetaData(attntrainedcount)
            settings = settings.get_settings_EEG_prepost()

            # iterate through subjects for individual subject analyses

            for sub_count, sub_val in enumerate(settings.subsIDXcollate):
                # get directories and file names
                bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
                print(bids.substring)

                # sort by behaviour
                subcondstring = attntrained + str(sub_val)
                if any(highperformers.isin([subcondstring])):
                    sub_count_high = sub_count_high + 1

                    # load results
                    results = np.load(bids.direct_results / Path(bids.substring + "RSA_data.npz"), allow_pickle=True) #, RDM=RDM, SSVEPS_RSA=SSVEPS_RSA)
                    SSVEPS_RSA_high[:, :, :, sub_count_high] = results['SSVEPS_RSA']

        # Collate data
        # subtract mean value from each electrode and frequency so we're just targetting the variance across conditions.
        SSVEPS_RSA_high = np.mean(SSVEPS_RSA_high, axis=2)
        SSVEPS_RSA_high = np.divide(SSVEPS_RSA_high - SSVEPS_RSA_high.mean(axis=1)[:, None, :], SSVEPS_RSA_high.std(axis=1)[:, None, :])

        SSVEPS_RSA_Group_high = SSVEPS_RSA_high.mean(axis=2)
        SSVEPS_RSA_Groupuse = SSVEPS_RSA_Group_high

        # Run RSA on group data
        RDM_Group_HP = np.zeros((settings.num_conditionsRSA, settings.num_conditionsRSA))
        RDM_Group_HP[:] = np.nan

        for conditionA in np.arange(settings.num_conditionsRSA):
            for conditionB in np.arange(settings.num_conditionsRSA):
                corr = stats.spearmanr(SSVEPS_RSA_Groupuse[:, conditionA], SSVEPS_RSA_Groupuse[:, conditionB])
                correlation_distance = 1 - corr[0]
                RDM_Group_HP[conditionA, conditionB] = correlation_distance

        fig, ax1 = plt.subplots(1, 1, figsize=(6,6))
        im = ax1.imshow(RDM_Group_HP[:, :], clim=(0.0, 2))
        plt.set_cmap('jet')
        titlestring = attntrained + ' RSA High Performing Participants ' + cuetypestring
        fig.suptitle(titlestring)
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

        highperformersRDM = np.array((np.nan))
        for i in np.arange(settings.num_conditionsRSA):
            highperformersRDM = np.hstack((highperformersRDM, RDM_Group_HP[i, i + 1:]))




        ########## Compare those high performers to everyone else.
        # list objects for bootstrapped group RSA
        TrainingGroup_grp = list()
        Bootstrap_grp = list()
        RDM_Score_D1 = list()
        RDM_Score_D4 = list()
        Testday_grp = list()
        RDM_Score = list()

        # cycle trough space and feature train groups
        for attntrainedcount, attntrained in enumerate(settings.string_attntrained):
            # setup generic settings
            settings = helper.SetupMetaData(attntrainedcount)
            settings = settings.get_settings_EEG_prepost()

            ## preallocate for group
            settings.num_Hz = 4
            SSVEPS_RSA = np.zeros((settings.num_electrodes * settings.num_Hz + settings.num_electrodes *2, settings.num_conditionsRSA, settings.num_days, settings.num_subs))
            SSVEPS_RSA[:] = np.nan

            # iterate through subjects for individual subject analyses
            for sub_count, sub_val in enumerate(settings.subsIDXcollate):
                # get directories and file names
                bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
                print(bids.substring)

                # sort by behaviour
                subcondstring = attntrained + str(sub_val)
                if not(any(highperformers.isin([subcondstring]))):
                    # load results
                    results = np.load(bids.direct_results / Path(bids.substring + "RSA_data.npz"), allow_pickle=True)  # , RDM=RDM, SSVEPS_RSA=SSVEPS_RSA)
                    SSVEPS_RSA[:, :, :, sub_count] = results['SSVEPS_RSA']

            # Collate data
            # subtract mean value from each electrode and frequency so we're just targetting the variance across conditions.
            SSVEPS_RSA[:, :, 0, :] = np.divide(SSVEPS_RSA[:, :, 0, :] - np.mean(SSVEPS_RSA[:, :, 0, :], axis=1)[:, None, :], SSVEPS_RSA[:, :, 0, :].std(axis=1)[:, None, :])
            SSVEPS_RSA[:, :, 1, :] = np.divide(SSVEPS_RSA[:, :, 1, :] - np.mean(SSVEPS_RSA[:, :, 1, :], axis=1)[:, None, :], SSVEPS_RSA[:, :, 1, :].std(axis=1)[:, None, :])

            SSVEPS_RSA_Group = np.nanmean(SSVEPS_RSA, axis=3)
            SSVEPS_RSA_Groupuse = SSVEPS_RSA_Group
            # Run RSA on group data
            RDM_Group = np.zeros((settings.num_conditionsRSA, settings.num_conditionsRSA, settings.num_days))
            RDM_Group[:] = np.nan
            for day_count, day_val in enumerate(settings.daysuse):
                for conditionA in np.arange(settings.num_conditionsRSA):
                    for conditionB in np.arange(settings.num_conditionsRSA):
                        corr = stats.spearmanr(SSVEPS_RSA_Groupuse[:, conditionA, day_count], SSVEPS_RSA_Groupuse[:, conditionB, day_count])
                        correlation_distance = 1 - corr[0]
                        RDM_Group[conditionA, conditionB, day_count] = correlation_distance

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
            im = ax1.imshow(RDM_Group[:, :, 0], clim=(0.0, 2))
            ax1.set_title('Pre-Training')
            im = ax2.imshow(RDM_Group[:, :, 1], clim=(0.0, 2))
            ax2.set_title('Post-Training')
            plt.set_cmap('jet')
            titlestring = attntrained + ' RSA '  + cuetypestring
            fig.suptitle(titlestring)
            # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
            # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

            day1 = np.array((np.nan))
            day4 = np.array((np.nan))
            for i in np.arange(settings.num_conditionsRSA):
                day1 = np.hstack((day1, RDM_Group[i, i + 1:, 0]))
                day4 = np.hstack((day4, RDM_Group[i, i + 1:, 1]))

            corr = stats.spearmanr(day1[1:], highperformersRDM[1:])
            print(1-corr[0])
            corr = stats.spearmanr(day4[1:], highperformersRDM[1:])
            print(1 - corr[0])

            ## Bootstrap analysis
            # Run RSA on group data
            num_bootstraps = 100
            for bootstrap in np.arange(num_bootstraps):
                # work out Representational dissimilarity across days
                conduse = np.random.choice(settings.num_conditionsRSA, settings.num_conditionsRSA, replace=True)
                # conduse = np.arange(32)

                datuse = RDM_Group[:, conduse, :]
                datuse_HP = RDM_Group_HP[:, conduse]

                day1 = np.array((np.nan))
                day4 = np.array((np.nan))
                HP = np.array((np.nan))
                for i in np.arange(settings.num_conditionsRSA):
                    day1 = np.hstack((day1, datuse[i, i + 1:, 0]))
                    day4 = np.hstack((day4, datuse[i, i + 1:, 1]))
                    HP = np.hstack((HP, datuse_HP[i, i + 1:]))

                nans = np.logical_or(np.logical_or(day1 ==0, day4==0), HP ==0)
                day1 = np.delete(day1, nans)
                day4 = np.delete(day4, nans)
                HP = np.delete(HP, nans)

                corrd1 = stats.spearmanr(day1[1:], HP[1:])
                corrd4 = stats.spearmanr(day4[1:], HP[1:])

                for testday in ['pre-training', 'post-training']:
                    TrainingGroup_grp.append(attntrained)
                    Bootstrap_grp.append(bootstrap)
                    RDM_Score_D1.append(1 - corrd1[0])
                    RDM_Score_D4.append(1 - corrd4[0])
                    Testday_grp.append(testday)
                    if testday == 'pre-training':
                        RDM_Score.append(1 - corrd1[0])
                    else:
                        RDM_Score.append(1 - corrd4[0])

        # create dataframe of group results
        data = {'TrainingGroup': TrainingGroup_grp, 'Bootstrap': Bootstrap_grp, 'Testday': Testday_grp, 'RDM_ScoreD1': RDM_Score_D1, 'RDM_ScoreD4': RDM_Score_D4, 'RDM_Score': RDM_Score }
        df_RDM_grp = pd.DataFrame(data)

        # stats n stuff.
        df_RDM_grp.groupby('TrainingGroup').mean()
        stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Feature"]), 'RDM_ScoreD1'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Feature"]), 'RDM_ScoreD4'])
        stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Sham"]), 'RDM_ScoreD1'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Sham"]), 'RDM_ScoreD4'])
        stats.ttest_ind(df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Space"]), 'RDM_ScoreD1'], df_RDM_grp.loc[df_RDM_grp.TrainingGroup.isin(["Space"]), 'RDM_ScoreD4'])

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        colors = [settings.yellow, settings.orange, settings.red]
        # sns.swarmplot(x="TrainingGroup", y="RDM_Score", hue='Testday', data=df_RDM_grp, color="0", alpha=0.3, ax=ax, dodge=True)
        sns.violinplot(x="TrainingGroup", y="RDM_Score", hue='Testday',data=df_RDM_grp, palette=sns.color_palette(colors), style="ticks",
                       ax=ax, inner="box", alpha=0.6)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel('Distance from pre-training')
        titlestring = 'RSA pre to post training distance ' + cuetypestring
        plt.suptitle(titlestring)
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
        plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

        # Get effects
        df_RDM_grp['RDM_Score_effect'] = df_RDM_grp['RDM_ScoreD4'] - df_RDM_grp['RDM_ScoreD1']
        df_trainingtmp = df_RDM_grp.loc[df_RDM_grp['Testday'] == 'pre-training', ['RDM_Score_effect', 'TrainingGroup']].copy()
        df_trainingtmp['Cuetype'] = cuetypestring
        if cuetype == 0:
            df_training = df_trainingtmp.copy()
        else:
            df_training = pd.concat([df_training, df_trainingtmp], axis=0)

    # Plot RSA training effect
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    colors = [settings.yellow, settings.orange, settings.red]
    sns.violinplot(x="TrainingGroup", y="RDM_Score_effect", hue='Cuetype', data=df_training, palette=sns.color_palette(colors), style="ticks",
                   ax=ax, inner="box", alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylabel('Distance from pre-training (post - pre)')
    ax.set_ylim([-0.15, 0.15])
    titlestring = 'RSA training distance training effect all conditions'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    colors = [settings.lightteal, settings.darkteal]
    df_trainingplot = df_training.loc[df_training['TrainingGroup'].isin(["Space", "Feature"])]
    sns.violinplot(x="TrainingGroup", y="RDM_Score_effect", hue='Cuetype', data=df_trainingplot, palette=sns.color_palette(colors), style="ticks",
                   ax=ax, inner="box", alpha=0.6)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.axhline(0, color='k', linestyle='--')
    ax.set_ylabel('Distance from pre-training (post - pre)')
    ax.set_ylim([-0.15, 0.15])
    titlestring = 'RSA training distance training effect Neurofeedback participants'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')