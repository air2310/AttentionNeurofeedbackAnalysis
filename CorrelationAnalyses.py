import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import helperfunctions_ATTNNF as helper
import h5py
import numpy as np
from pathlib import Path

def load_classifierdata(settings):
    ##### Load Classification Accuracy Data ####
    attntrained_vec = []
    traingroup_vec = []
    sub_vec = []
    classifiertype_vec = []
    classifieracc_vec = []

    # Cycle through trained groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_behave_prepost()  # get task specific settings

        if attntrained == 'Sham':  # get data for the feedback that was given
            # decide which file to use
            possiblefiles = []
            for filesfound in settings.direct_dataroot.glob("matched_sample.mat"):
                possiblefiles.append(filesfound)
            file2use = possiblefiles[0]

            # load data
            F = h5py.File(file2use, 'r')  # print(list(F.keys()))
            matchedsample = np.array(F['matchedsample'])
            matchedsample_traintype = abs(np.array(F['matchedsample_traintype']) - 2)

        # Cycle through subjects
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get file names
            if attntrained == 'Sham':
                sub_val_use = int(matchedsample[sub_val - 1][0])
                settings = helper.SetupMetaData(int(matchedsample_traintype[sub_val - 1][0]))
                settings = settings.get_settings_behave_prepost()
            else:
                sub_val_use = sub_val
            bids = helper.BIDS_FileNaming(sub_val_use, settings, day_val=1)

            # Get Data for Space and Feature Classifier
            for attn_count, attn_val in enumerate(['Space', 'Feature']):
                # decide which file to use
                possiblefiles = []
                for filesfound in bids.direct_data_eeg.glob(
                        "Classifier_" + attn_val + '_' + bids.filename_eeg + ".mat"):
                    possiblefiles.append(filesfound)
                file2use = possiblefiles[0]

                # load data
                F = h5py.File(file2use, 'r')  # print(list(F.keys()))

                # get Accuracy
                tmp_acc = np.array(F['ACCURACY_ALL']) * 100

                if attntrained == 'Sham':
                    attntrained_vec.append(settings.string_attntrained[int(matchedsample_traintype[sub_val - 1][0])])
                else:
                    attntrained_vec.append(attntrained)
                traingroup_vec.append(attntrained)
                sub_vec.append(attntrained + str(sub_val))
                classifiertype_vec.append(attn_val)
                classifieracc_vec.append(np.nanmean(tmp_acc))

    # Stack data into a dataframe
    data = {'SubID': sub_vec, 'TrainingGroup': traingroup_vec, 'AttentionTrained': attntrained_vec, 'ClassifierType': classifiertype_vec,
            'ClassifierAccuracy': classifieracc_vec}
    df_classifier = pd.DataFrame(data)

    df_classifier_condensed = df_classifier.loc[df_classifier.AttentionTrained == df_classifier.ClassifierType, :].copy()
    df_classifier_condensed = df_classifier_condensed.reset_index().drop(columns=['ClassifierType', 'index']).copy()

    return df_classifier, df_classifier_condensed


def load_SSVEPdata(settings):

    # preallocate
    select_pre_space = list()
    select_pre_feature = list()
    select_post_space = list()
    select_post_feature = list()
    select_train_space = list()
    select_train_feature = list()
    traingroup_vec = list()
    sub_vec = list()

    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_EEG_prepost()

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 1)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "EEG_pre_post_results.npz"), allow_pickle=True)
            # saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

            # store results
            # tmp = results['SSVEPs_prepost_channelmean_epochs']
            tmp = results['SSVEPs_prepost_channelmean'] #settings.num_attd_unattd, settings.num_days, settings.num_attnstates
            selectivity = tmp[0, :, :] - tmp[1, :, :]

            select_pre_space.append(selectivity[0,0])
            select_pre_feature.append(selectivity[0, 1])
            select_post_space.append(selectivity[1, 0])
            select_post_feature.append(selectivity[1, 1])
            select_train_space.append(selectivity[1, 0] - selectivity[0, 0])
            select_train_feature.append(selectivity[1, 1] - selectivity[0, 1])

            traingroup_vec.append(attntrained)
            sub_vec.append(attntrained + str(sub_val))

    data = {'SubID': sub_vec, 'TrainingGroup': traingroup_vec, 'select_pre_space': select_pre_space,
            'select_pre_feature': select_pre_feature, 'select_post_space': select_post_space, 'select_post_feature': select_post_feature,
            'select_train_space': select_train_space, 'select_train_feature': select_train_feature}

    df_selectivity = pd.DataFrame(data)

    return df_selectivity


def load_MotionDiscrimBehaveResults(settings):

    # initialise
    sensitivity_pre_spacecue = list()
    sensitivity_post_spacecue = list()
    sensitivity_train_spacecue = list()
    sensitivity_pre_featcue = list()
    sensitivity_post_featcue = list()
    sensitivity_train_featcue = list()
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

            traingroup_vec.append(attntrained)
            sub_vec.append(attntrained + str(sub_val))

    data = {'SubID': sub_vec, 'TrainingGroup': traingroup_vec,
            'sensitivity_pre_spacecue': sensitivity_pre_spacecue, 'sensitivity_post_spacecue': sensitivity_post_spacecue, 'sensitivity_train_spacecue': sensitivity_train_spacecue,
            'sensitivity_pre_featcue': sensitivity_pre_featcue, 'sensitivity_post_featcue': sensitivity_post_featcue, 'sensitivity_train_featcue': sensitivity_train_featcue}

    df_sensitivity = pd.DataFrame(data)

    return df_sensitivity, bids


def plot_classificationacc(df_classifier_condensed, bids, settings):
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow, settings.orange]

    # Accuracy Grouped violinplot
    sns.swarmplot(x="AttentionTrained", y="ClassifierAccuracy", data=df_classifier_condensed[df_classifier_condensed['TrainingGroup'] != "Sham"], color=".5")
    sns.violinplot(x="AttentionTrained", y="ClassifierAccuracy", data=df_classifier_condensed[df_classifier_condensed['TrainingGroup'] != "Sham"], palette=sns.color_palette(colors), style="ticks", ax=ax)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim(0, 100)
    ax.set_title("Accuracy")

    titlestring = 'Classification Accuracy'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


def classification_acc_correlations(settings):
    import scipy.stats as stats

    # Get data
    df_classifier, df_classifier_condensed = load_classifierdata(settings)
    df_selectivity = load_SSVEPdata(settings)
    df_sensitivity, bids = load_MotionDiscrimBehaveResults(settings)

    # plot classifier accuracy by attention type
    plot_classificationacc(df_classifier_condensed, bids, settings)

    # Correlations!

    # Sensitivity
    datuse = pd.concat([df_classifier_condensed['TrainingGroup'], df_classifier_condensed['ClassifierAccuracy'],  df_sensitivity['sensitivity_train_spacecue'], df_sensitivity['sensitivity_train_featcue']], axis=1)
    datuse = datuse.loc[datuse.TrainingGroup.isin(['Space']), ].copy()
    corrs_space = stats.pearsonr(datuse['ClassifierAccuracy'], datuse['sensitivity_train_spacecue'])
    corrs_feat = stats.pearsonr(datuse['ClassifierAccuracy'], datuse['sensitivity_train_featcue'])

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]
    datuse = pd.concat([df_classifier_condensed['TrainingGroup'], df_classifier_condensed['ClassifierAccuracy'], df_sensitivity['sensitivity_train_spacecue'], df_sensitivity['sensitivity_train_featcue']], axis=1)
    sns.scatterplot(data=datuse.loc[datuse.TrainingGroup.isin(['Space']), ], x="ClassifierAccuracy", y='sensitivity_train_spacecue', ax=ax, color=settings.yellow_)
    sns.scatterplot(data=datuse.loc[datuse.TrainingGroup.isin(['Feature']), ], x="ClassifierAccuracy", y='sensitivity_train_spacecue', ax=ax, color=settings.lightteal_)

    ax.set_title("r = " + str(corrs_space[0]) + ', p = ' + str(corrs_space[1]))
    ax.set_ylabel("Selectivity for trained attention type")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    titlestring = "Classifier Acc Vs. Selectivity Scatter"
    plt.suptitle(titlestring)


    # plot classification accuracy vs. Selectivity


    corrs_space = stats.pearsonr(df_classifier_condensed['ClassifierAccuracy'], df_selectivity['select_pre_space'])
    corrs_feat = stats.pearsonr(df_classifier_condensed['ClassifierAccuracy'], df_selectivity['select_pre_feature'])


    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    tmp = pd.concat([df_classifier_condensed['TrainingGroup'], df_classifier_condensed['ClassifierAccuracy'], df_selectivity['select_pre_feature']], axis=1)

    sns.scatterplot(data=tmp.loc[tmp.TrainingGroup.isin(['Space']),], x="ClassifierAccuracy", y='select_pre_feature', ax=ax, color=settings.yellow_)
    sns.scatterplot(data=tmp.loc[tmp.TrainingGroup.isin(['Feature']), ], x="ClassifierAccuracy", y='select_pre_feature', ax=ax, color=settings.lightteal_)

    ax.set_title("r = " + str(corrs_feat[0]) + ', p = ' + str(corrs_feat[1]))
    ax.set_ylabel("Selectivity for trained attention type")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    titlestring = "Classifier Acc Vs. Selectivity Scatter"
    plt.suptitle(titlestring)
    # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')



def classification_acc_correlations_old(settings):

    ##### Load Classification Accuracy Data ####
    attntrained_vec = []
    sub_vec = []
    classifiertype_vec = []
    classifieracc_vec = []

    # Cycle through trained groups
    for attntrainedcount, attntrained in enumerate(['Space', 'Feature']):
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)

        # get task specific settings
        settings = settings.get_settings_behave_prepost()

        # Cycle through subjects
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get file names
            bids = helper.BIDS_FileNaming(sub_val, settings, day_val=1)

            # Get Data for Space and Feature Classifier
            for attn_count, attn_val in enumerate(['Space', 'Feature']):
                # decide which file to use
                possiblefiles = []
                for filesfound in bids.direct_data_eeg.glob("Classifier_" + attn_val + '_' + bids.filename_eeg + ".mat"):
                    possiblefiles.append(filesfound)
                file2use = possiblefiles[0]

                # load data
                F = h5py.File(file2use, 'r')
                # print(list(F.keys()))

                # get Accuracy
                tmp_acc = np.array(F['ACCURACY_ALL']) * 100

                attntrained_vec.append(attntrained)
                sub_vec.append(attntrained + str(sub_val))
                classifiertype_vec.append(attn_val)
                classifieracc_vec.append(np.nanmean(tmp_acc))

    # Stack data into a dataframe
    data = {'SubID': sub_vec, 'AttentionTrained': attntrained_vec, 'ClassifierType': classifiertype_vec,
            'ClassifierAccuracy': classifieracc_vec}
    df_classifier = pd.DataFrame(data)

    df_classifier_condensed = df_classifier.loc[df_classifier.AttentionTrained == df_classifier.ClassifierType,
                              :].copy()
    # plot results

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.medteal_, settings.lightteal_]

    # Accuracy Grouped violinplot
    sns.violinplot(x="AttentionTrained", y="ClassifierAccuracy", data=df_classifier_condensed,
                   palette=sns.color_palette(colors), style="ticks", ax=ax)
    sns.swarmplot(x="AttentionTrained", y="ClassifierAccuracy", data=df_classifier_condensed, color=".5")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim(0, 100)
    ax.set_title("Accuracy")

    titlestring = 'Classification Accuracy'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # plot results - both used and unused
    df_classifier_used = df_classifier.loc[df_classifier.AttentionTrained == df_classifier.ClassifierType, :].copy()
    df_classifier_unused = df_classifier.loc[df_classifier.AttentionTrained != df_classifier.ClassifierType, :].copy()

    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    sns.set(style="ticks")
    colors = [settings.medteal_, settings.lightteal_]

    # Accuracy Grouped violinplot
    sns.violinplot(x="ClassifierType", y="ClassifierAccuracy", data=df_classifier,
                   palette=sns.color_palette(colors), style="ticks", ax=ax)
    sns.swarmplot(x="ClassifierType", y="ClassifierAccuracy", data=df_classifier_used, color="0.5")
    sns.swarmplot(x="ClassifierType", y="ClassifierAccuracy", data=df_classifier_unused, color="0.5",
                  order=['Space', 'Feature'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim(0, 100)
    ax.set_title("Accuracy")

    titlestring = 'Classification Accuracy used vs unused'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    df_classifier[df_classifier['ClassifierType'] == 'Space'].mean()  # 65.744799
    df_classifier[df_classifier['ClassifierType'] == 'Space'].std()  # 12.483115

    df_classifier[df_classifier['ClassifierType'] == 'Feature'].mean()  # 54.375419
    df_classifier[df_classifier['ClassifierType'] == 'Feature'].std()  # 5.715636
    ########## Get SSVEP Selectivity Data #############

    print('collating SSVEP amplitudes pre Vs. post training comparing Space Vs. Feat Training')

    # preallocate
    num_subs = np.zeros((settings.num_attnstates))
    daystrings = []
    attnstrings = []
    attntaskstrings = []
    selectivity_compare = []

    # cycle trough space and feature train groups
    for attntrained in np.arange(settings.num_attnstates):  # cycle trough space and feature train groups

        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_EEG_prepost()

        # file names
        bids = helper.BIDS_FileNaming(subject_idx=0, settings=settings, day_val=0)

        # load results
        results = np.load(bids.direct_results_group / Path("EEGResults_prepost.npz"), allow_pickle=True)  #
        # settings.num_subs =
        SSVEPs_epochs_prepost_group = results['SSVEPs_epochs_prepost_group']  # results['SSVEPs_prepost_group']
        diffdat = SSVEPs_epochs_prepost_group[0, :, :, :] - SSVEPs_epochs_prepost_group[1, :, :, :]  # [day,attn,sub]

        # store results for attention condition
        tmp = [settings.string_prepost[0]] * settings.num_subs * settings.num_attnstates + [
            settings.string_prepost[1]] * settings.num_subs * settings.num_attnstates
        daystrings = np.concatenate((daystrings, tmp))

        tmp = [settings.string_attntrained[0]] * settings.num_subs + [
            settings.string_attntrained[1]] * settings.num_subs
        attntaskstrings = np.concatenate((attntaskstrings, tmp, tmp))

        tmp = [settings.string_attntrained[
                   attntrained]] * settings.num_subs * settings.num_days * settings.num_attnstates
        attnstrings = np.concatenate((attnstrings, tmp))

        tmp = np.concatenate((diffdat[0, 0, :], diffdat[0, 1, :], diffdat[1, 0, :], diffdat[1, 1, :]))
        selectivity_compare = np.concatenate((selectivity_compare, tmp))

    data = {'Testday': daystrings, 'Attention Type': attntaskstrings, 'Attention Trained': attnstrings,
            'Selectivity (ΔµV)': selectivity_compare}
    df_selectivity = pd.DataFrame(data)

    # Add selectivity - get the indices we're interested in.
    idx_space_pre = np.logical_and(df_selectivity.Testday == 'pre-training', df_selectivity["Attention Type"] == "Space")
    idx_feature_pre = np.logical_and(df_selectivity.Testday == 'pre-training',
                                     df_selectivity["Attention Type"] == "Feature")

    idx_space_post = np.logical_and(df_selectivity.Testday == 'post-training',
                                    df_selectivity["Attention Type"] == "Space")
    idx_feature_post = np.logical_and(df_selectivity.Testday == 'post-training',
                                      df_selectivity["Attention Type"] == "Feature")

    # create new correlation dataframe to add selectivity to
    df_correlationsdat = df_classifier_condensed.drop(["ClassifierType"], axis=1).copy().reset_index()
    df_correlationsdat = df_correlationsdat.drop(["index"], axis=1)

    # add selectivity - pre and post training
    df_correlationsdat["Space_Selectivity_pre"] = df_selectivity.loc[idx_space_pre, :].reset_index().loc[:,
                                                  "Selectivity (ΔµV)"]
    df_correlationsdat["Feature_Selectivity_pre"] = df_selectivity.loc[idx_feature_pre, :].reset_index().loc[:,
                                                    "Selectivity (ΔµV)"]

    df_correlationsdat["Space_Selectivity_post"] = df_selectivity.loc[idx_space_post, :].reset_index().loc[:,
                                                   "Selectivity (ΔµV)"]
    df_correlationsdat["Feature_Selectivity_post"] = df_selectivity.loc[idx_feature_post, :].reset_index().loc[:,
                                                     "Selectivity (ΔµV)"]

    # Add training effect
    df_correlationsdat["Space_Selectivity_trainefc"] = df_selectivity.loc[idx_space_post, :].reset_index().loc[:,
                                                       "Selectivity (ΔµV)"] - df_selectivity.loc[idx_space_pre,
                                                                              :].reset_index().loc[:,
                                                                              "Selectivity (ΔµV)"]
    df_correlationsdat["Feature_Selectivity_trainefc"] = df_selectivity.loc[idx_feature_post, :].reset_index().loc[:,
                                                         "Selectivity (ΔµV)"] - df_selectivity.loc[idx_feature_pre,
                                                                                :].reset_index().loc[:,
                                                                                "Selectivity (ΔµV)"]

    ######## Load behavioural data
    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained[0:2]):  # cycle trough space and feature train groups
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_behave_prepost()

        # file names
        bids = helper.BIDS_FileNaming(subject_idx=0, settings=settings, day_val=0)

        # accdat_all_avg.to_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))
        df_behaveresults_tmp = pd.read_pickle(
            bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))
        df_behaveresults_tmp['AttentionTrained'] = attntrained

        if (attntrainedcount == 0):
            df_behaveresults = df_behaveresults_tmp[
                ['AttentionTrained', 'Testday', 'Attention Type', 'Sensitivity', 'Criterion', 'RT']]
        else:
            df_behaveresults = df_behaveresults.append(df_behaveresults_tmp[
                                                           ['AttentionTrained', 'Testday', 'Attention Type',
                                                            'Sensitivity', 'Criterion', 'RT']])

    # Add data to correlations dat - get indices
    idx_space_pre = np.logical_and(df_behaveresults.Testday == 'Day 1', df_behaveresults["Attention Type"] == "Space")
    idx_feature_pre = np.logical_and(df_behaveresults.Testday == 'Day 1',
                                     df_behaveresults["Attention Type"] == "Feature")

    idx_space_post = np.logical_and(df_behaveresults.Testday == 'Day 4', df_behaveresults["Attention Type"] == "Space")
    idx_feature_post = np.logical_and(df_behaveresults.Testday == 'Day 4',
                                      df_behaveresults["Attention Type"] == "Feature")

    # Add data to correlations dat
    dattype = "Sensitivity"
    df_correlationsdat["Space_" + dattype + "_pre"] = df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:,
                                                      dattype]
    df_correlationsdat["Feature_" + dattype + "_pre"] = df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:,
                                                        dattype]

    df_correlationsdat["Space_" + dattype + "_post"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:,
                                                       dattype]
    df_correlationsdat["Feature_" + dattype + "_post"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:,
                                                         dattype]

    # Add training effect
    df_correlationsdat["Space_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:,
                                                           dattype] - df_behaveresults.loc[idx_space_pre,
                                                                      :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_feature_post,
                                                             :].reset_index().loc[:, dattype] - df_behaveresults.loc[
                                                                                                idx_feature_pre,
                                                                                                :].reset_index().loc[:,
                                                                                                dattype]

    # Add data to correlations dat
    dattype = "Criterion"
    df_correlationsdat["Space_" + dattype + "_pre"] = df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:,
                                                      dattype]
    df_correlationsdat["Feature_" + dattype + "_pre"] = df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:,
                                                        dattype]

    df_correlationsdat["Space_" + dattype + "_post"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:,
                                                       dattype]
    df_correlationsdat["Feature_" + dattype + "_post"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:,
                                                         dattype]

    # Add training effect
    df_correlationsdat["Space_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:,
                                                           dattype] - df_behaveresults.loc[idx_space_pre,
                                                                      :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_feature_post,
                                                             :].reset_index().loc[:, dattype] - df_behaveresults.loc[
                                                                                                idx_feature_pre,
                                                                                                :].reset_index().loc[:,
                                                                                                dattype]

    # Add data to correlations dat
    dattype = "RT"
    df_correlationsdat["Space_" + dattype + "_pre"] = df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:,
                                                      dattype]
    df_correlationsdat["Feature_" + dattype + "_pre"] = df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:,
                                                        dattype]

    df_correlationsdat["Space_" + dattype + "_post"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:,
                                                       dattype]
    df_correlationsdat["Feature_" + dattype + "_post"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:,
                                                         dattype]

    # Add training effect
    df_correlationsdat["Space_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:,
                                                           dattype] - df_behaveresults.loc[idx_space_pre,
                                                                      :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_feature_post,
                                                             :].reset_index().loc[:, dattype] - df_behaveresults.loc[
                                                                                                idx_feature_pre,
                                                                                                :].reset_index().loc[:,
                                                                                                dattype]

    # plot classification accuracy vs. Selectivity
    import scipy.stats as stats

    i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_Selectivity_pre"]]
    k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_Selectivity_pre"]]
    corrs_space = stats.pearsonr(i["ClassifierAccuracy"], j["Space_Selectivity_pre"])
    corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_Selectivity_pre"])
    corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"], k["ClassifierAccuracy"]], axis=0),
                                pd.concat([j["Space_Selectivity_pre"], l["Feature_Selectivity_pre"]], axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="ClassifierAccuracy",
                    y="Space_Selectivity_pre", ax=ax, color=settings.yellow_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy",
                    y="Feature_Selectivity_pre", ax=ax, color=settings.lightteal_)
    ax.set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    ax.set_ylabel("Selectivity for trained attention type")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    titlestring = "Classifier Acc Vs. Selectivity Scatter"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Plot how training effects relate to classifcation accuracy

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    # Selectivity
    i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_Selectivity_trainefc"]]
    k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_Selectivity_trainefc"]]
    corrs_space = stats.pearsonr(i["ClassifierAccuracy"], j["Space_Selectivity_trainefc"])
    corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_Selectivity_trainefc"])
    corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"], k["ClassifierAccuracy"]], axis=0),
                                pd.concat([j["Space_Selectivity_trainefc"], l["Feature_Selectivity_trainefc"]], axis=0))

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy",
                    y="Feature_Selectivity_trainefc", ax=ax[0], color=settings.lightteal_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="ClassifierAccuracy",
                    y="Space_Selectivity_trainefc", ax=ax[0], color=settings.yellow_)
    ax[0].set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    # ax[0].set_title("Classifier Acc Vs. Selectivity train effect")
    ax[0].set_ylabel("change in Selectivity for trained attention type")
    ax[0].legend(['Feature', 'Space'], title="Attention Trained")
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    # Sensitivity
    i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_Sensitivity_trainefc"]]
    k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_Sensitivity_trainefc"]]
    corrs_space = stats.pearsonr(i["ClassifierAccuracy"], j["Space_Sensitivity_trainefc"])
    corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_Sensitivity_trainefc"])
    corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"], k["ClassifierAccuracy"]], axis=0),
                                pd.concat([j["Space_Sensitivity_trainefc"], l["Feature_Sensitivity_trainefc"]], axis=0))

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy",
                    y="Feature_Sensitivity_trainefc", ax=ax[1], color=settings.lightteal_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="ClassifierAccuracy",
                    y="Space_Sensitivity_trainefc", ax=ax[1], color=settings.yellow_)
    ax[1].set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    # ax[1].set_title("Classifier Acc Vs. Sensitivity (d') train effect")
    ax[1].set_ylabel("Change in Sensitivity for trained attention type")
    ax[1].legend(['Feature', 'Space'], title="Attention Trained")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    titlestring = "Effect of classifier accuracy on training"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # plot Retest Reliability of measures
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=df_correlationsdat, x="Space_Selectivity_pre", y="Space_Selectivity_post",
                    hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[0][0])
    ax[0][0].set_title("Space Selectivity Pre Vs. Post")

    sns.scatterplot(data=df_correlationsdat, x="Feature_Selectivity_pre", y="Feature_Selectivity_post",
                    hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[0][1])
    ax[0][1].set_title("Feature Selectivity Pre Vs. Post")

    sns.scatterplot(data=df_correlationsdat, x="Space_Sensitivity_pre", y="Space_Sensitivity_post",
                    hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[1][0])
    ax[1][0].set_title("Space Sensitivity Pre Vs. Post")

    sns.scatterplot(data=df_correlationsdat, x="Feature_Sensitivity_pre", y="Feature_Sensitivity_post",
                    hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[1][1])
    ax[1][1].set_title("Feature Selectivity Pre Vs. Post")

    titlestring = "Retest reliability of attention measures"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Plot Construct validity

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="Space_Selectivity_pre",
                    y="Space_Sensitivity_pre", ax=ax[0], color=settings.yellow_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"],
                    x="Feature_Selectivity_pre", y="Feature_Sensitivity_pre", ax=ax[0], color=settings.lightteal_)
    ax[0].set_title("SSVEP Selectivity Vs. Behave Sensitivity (pre train)")
    ax[0].set_ylabel("Sensitivity (d')")
    ax[0].set_xlabel("SSVEP Selectivity")
    ax[0].legend(settings.string_attntrained, title="Attention Trained")
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    sns.scatterplot(data=df_correlationsdat, x="Space_Selectivity_pre", y="Feature_Selectivity_pre",
                    hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[1])
    ax[1].set_title("Selectivity Space Vs Feature (pre train)")
    ax[1].set_ylabel("Feature_Selectivity_pre")
    ax[1].set_xlabel("Space_Selectivity_pre")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    sns.scatterplot(data=df_correlationsdat, x="Space_Sensitivity_pre", y="Feature_Sensitivity_pre",
                    hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[2])
    ax[2].set_title("Sensitivity Space Vs Feature (pre train)")
    ax[2].set_ylabel("Feature_Sensitivity_pre")
    ax[2].set_xlabel("Space_Sensitivity_pre")
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)

    titlestring = "Construct validity"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Does improvement on transfer tasks relate to improvement on training task?

    # Correct for missing transfer task data from subjects 89 and 90 and NaN for subject 116
    df_correlationsdat_transfertasks = df_correlationsdat.loc[~df_correlationsdat.SubID.isin([89, 90, 116])]

    # Load visual search data
    vissearchrt = pd.read_pickle(bids.direct_results_group / Path("group_visualsearch.pkl"))

    # Add data to correlations dat - get indices
    idx_space_pre_VS = np.logical_and(vissearchrt.Testday == 'pre-training',
                                      vissearchrt["Attention Trained"] == "Space")
    idx_feature_pre_VS = np.logical_and(vissearchrt.Testday == 'pre-training',
                                        vissearchrt["Attention Trained"] == "Feature")

    idx_space_post_VS = np.logical_and(vissearchrt.Testday == 'post-training',
                                       vissearchrt["Attention Trained"] == "Space")
    idx_feature_post_VS = np.logical_and(vissearchrt.Testday == 'post-training',
                                         vissearchrt["Attention Trained"] == "Feature")

    # calculate training effect
    vissearch_correlationsdat_Spacetrain = vissearchrt.loc[idx_space_post_VS, :].reset_index().loc[:,
                                           ["SubID", "Reaction Time (s)"]]
    vissearch_correlationsdat_Spacetrain["Reaction Time (s)"] = vissearchrt.loc[idx_space_post_VS, :].reset_index().loc[
                                                                :,
                                                                "Reaction Time (s)"] - vissearchrt.loc[idx_space_pre_VS,
                                                                                       :].reset_index().loc[:,
                                                                                       "Reaction Time (s)"]
    vissearch_correlationsdat_Spacetrain["Space_Sensitivity_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Space", "Space_Sensitivity_trainefc"]
    vissearch_correlationsdat_Spacetrain["Space_RT_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Space", "Space_RT_trainefc"]
    vissearch_correlationsdat_Spacetrain["Space_Criterion_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Space", "Space_Criterion_trainefc"]

    vissearch_correlationsdat_Feattrain = vissearchrt.loc[idx_feature_post_VS, :].reset_index().loc[:,
                                          ["SubID", "Reaction Time (s)"]]
    vissearch_correlationsdat_Feattrain["Reaction Time (s)"] = vissearchrt.loc[idx_feature_post_VS,
                                                               :].reset_index().loc[:,
                                                               "Reaction Time (s)"] - vissearchrt.loc[
                                                                                      idx_feature_pre_VS,
                                                                                      :].reset_index().loc[:,
                                                                                      "Reaction Time (s)"]
    vissearch_correlationsdat_Feattrain["Feature_Sensitivity_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Feature", "Feature_Sensitivity_trainefc"].reset_index().Feature_Sensitivity_trainefc
    vissearch_correlationsdat_Feattrain["Feature_RT_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Feature", "Feature_RT_trainefc"].reset_index().Feature_RT_trainefc
    vissearch_correlationsdat_Feattrain["Feature_Criterion_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Feature", "Feature_Criterion_trainefc"].reset_index().Feature_Criterion_trainefc

    # Correlate Sensitivity!

    i = vissearch_correlationsdat_Spacetrain["Reaction Time (s)"]
    j = df_correlationsdat_transfertasks.loc[df_correlationsdat_transfertasks.AttentionTrained == "Space", :]
    k = vissearch_correlationsdat_Feattrain["Reaction Time (s)"]
    l = df_correlationsdat_transfertasks.loc[df_correlationsdat_transfertasks.AttentionTrained == "Feature", :]
    corrs_spacetrain = stats.pearsonr(i, j["Space_Sensitivity_trainefc"])
    corrs_feattrain = stats.pearsonr(k, l["Feature_Sensitivity_trainefc"])

    corrs_both = stats.pearsonr(pd.concat([i, k], axis=0),
                                pd.concat([j["Space_Sensitivity_trainefc"], l["Feature_Sensitivity_trainefc"]], axis=0))

    # df_correlationsdat_transfertasks.columns

    # Plot Sensitivity
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=vissearch_correlationsdat_Spacetrain, y="Reaction Time (s)", x="Space_Sensitivity_trainefc",
                    ax=ax, color=settings.yellow_)
    sns.scatterplot(data=vissearch_correlationsdat_Feattrain, y="Reaction Time (s)", x="Feature_Sensitivity_trainefc",
                    ax=ax, color=settings.lightteal_)
    ax.set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    ax.set_xlabel("Sensitivity for trained attention type")
    ax.set_ylabel("Visual Search RT for trained group")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    titlestring = "VisualsearchRT Vs Motion Discrim Sensitivity"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Correlate RT!

    corrs_spacetrain = stats.pearsonr(i, j["Space_RT_trainefc"])
    corrs_feattrain = stats.pearsonr(k, l["Feature_RT_trainefc"])

    corrs_both = stats.pearsonr(pd.concat([i, k], axis=0),
                                pd.concat([j["Space_RT_trainefc"], l["Feature_RT_trainefc"]], axis=0))

    # df_correlationsdat_transfertasks.columns

    # Plot RT
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=vissearch_correlationsdat_Spacetrain, y="Reaction Time (s)", x="Space_RT_trainefc", ax=ax,
                    color=settings.yellow_)
    sns.scatterplot(data=vissearch_correlationsdat_Feattrain, y="Reaction Time (s)", x="Feature_RT_trainefc", ax=ax,
                    color=settings.lightteal_)
    ax.set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    ax.set_xlabel("RT for trained attention type")
    ax.set_ylabel("Visual Search RT for trained group")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    titlestring = "VisualsearchRT Vs Motion Discrim RT"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Correlate Criterion!

    corrs_spacetrain = stats.pearsonr(i, j["Space_Criterion_trainefc"])
    corrs_feattrain = stats.pearsonr(k, l["Feature_Criterion_trainefc"])

    corrs_both = stats.pearsonr(pd.concat([i, k], axis=0),
                                pd.concat([j["Space_Criterion_trainefc"], l["Feature_Criterion_trainefc"]], axis=0))

    # df_correlationsdat_transfertasks.columns

    # Plot RT
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=vissearch_correlationsdat_Spacetrain, y="Reaction Time (s)", x="Space_Criterion_trainefc",
                    ax=ax, color=settings.yellow_)
    sns.scatterplot(data=vissearch_correlationsdat_Feattrain, y="Reaction Time (s)", x="Feature_Criterion_trainefc",
                    ax=ax, color=settings.lightteal_)
    ax.set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    ax.set_xlabel("Criterion for trained attention type")
    ax.set_ylabel("Visual Search Criterion for trained group")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    titlestring = "VisualsearchRT Vs Motion Discrim Criterion"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Now check on the N-back task
    # Correct for missing transfer task data from subjects 89 and 90 and NaN for subject 116
    df_correlationsdat_transfertasks = df_correlationsdat.loc[~df_correlationsdat.SubID.isin([89, 90])]

    # Load visual search data
    Nback = pd.read_pickle(bids.direct_results_group / Path("group_Nback.pkl"))

    # Add data to correlations dat - get indices
    idx_space_pre_NB = np.logical_and(Nback.Testday == 'pre-training',
                                      Nback["Attention Trained"] == "Space")
    idx_feature_pre_NB = np.logical_and(Nback.Testday == 'pre-training',
                                        Nback["Attention Trained"] == "Feature")

    idx_space_post_NB = np.logical_and(Nback.Testday == 'post-training',
                                       Nback["Attention Trained"] == "Space")
    idx_feature_post_NB = np.logical_and(Nback.Testday == 'post-training',
                                         Nback["Attention Trained"] == "Feature")

    # calculate training effect
    NB_correlationsdat_Spacetrain = Nback.loc[idx_space_post_NB, :].reset_index().loc[:,
                                    ["SubID", "Reaction Time (s)", "Accuracy (%)"]]
    NB_correlationsdat_Spacetrain["Reaction Time (s)"] = Nback.loc[idx_space_post_NB, :].reset_index().loc[:,
                                                         "Reaction Time (s)"] - Nback.loc[idx_space_pre_NB,
                                                                                :].reset_index().loc[:,
                                                                                "Reaction Time (s)"]
    NB_correlationsdat_Spacetrain["Accuracy (%)"] = Nback.loc[idx_space_post_NB, :].reset_index().loc[:,
                                                    "Accuracy (%)"] - Nback.loc[idx_space_pre_NB,
                                                                      :].reset_index().loc[:, "Accuracy (%)"]
    NB_correlationsdat_Spacetrain["Space_Sensitivity_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Space", "Space_Sensitivity_trainefc"]
    NB_correlationsdat_Spacetrain["Space_RT_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Space", "Space_RT_trainefc"]
    NB_correlationsdat_Spacetrain["Space_Criterion_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Space", "Space_Criterion_trainefc"]

    NB_correlationsdat_Feattrain = Nback.loc[idx_feature_post_NB, :].reset_index().loc[:,
                                   ["SubID", "Reaction Time (s)", "Accuracy (%)"]]
    NB_correlationsdat_Feattrain["Reaction Time (s)"] = Nback.loc[idx_feature_post_NB, :].reset_index().loc[:,
                                                        "Reaction Time (s)"] - Nback.loc[
                                                                               idx_feature_pre_NB, :].reset_index().loc[
                                                                               :, "Reaction Time (s)"]
    NB_correlationsdat_Feattrain["Accuracy (%)"] = Nback.loc[idx_feature_post_NB, :].reset_index().loc[:,
                                                   "Accuracy (%)"] - Nback.loc[
                                                                     idx_feature_pre_NB, :].reset_index().loc[:,
                                                                     "Accuracy (%)"]
    NB_correlationsdat_Feattrain["Feature_Sensitivity_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Feature", "Feature_Sensitivity_trainefc"].reset_index().Feature_Sensitivity_trainefc
    NB_correlationsdat_Feattrain["Feature_RT_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Feature", "Feature_RT_trainefc"].reset_index().Feature_RT_trainefc
    NB_correlationsdat_Feattrain["Feature_Criterion_trainefc"] = df_correlationsdat_transfertasks.loc[
        df_correlationsdat_transfertasks.AttentionTrained == "Feature", "Feature_Criterion_trainefc"].reset_index().Feature_Criterion_trainefc

    # Calculate correlations - Sensitivity
    i = NB_correlationsdat_Spacetrain["Reaction Time (s)"]
    j = df_correlationsdat_transfertasks.loc[df_correlationsdat_transfertasks.AttentionTrained == "Space", :]
    k = NB_correlationsdat_Feattrain["Reaction Time (s)"]
    l = df_correlationsdat_transfertasks.loc[df_correlationsdat_transfertasks.AttentionTrained == "Feature", :]
    corrs_spacetrain = stats.pearsonr(i, j["Space_Sensitivity_trainefc"])
    corrs_feattrain = stats.pearsonr(k, l["Feature_Sensitivity_trainefc"])

    corrs_both = stats.pearsonr(pd.concat([i, k], axis=0),
                                pd.concat([j["Space_Sensitivity_trainefc"], l["Feature_Sensitivity_trainefc"]], axis=0))

    # df_correlationsdat_transfertasks.columns

    # Plot Sensitivity
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=NB_correlationsdat_Spacetrain, y="Reaction Time (s)", x="Space_Sensitivity_trainefc", ax=ax,
                    color=settings.yellow_)
    sns.scatterplot(data=NB_correlationsdat_Feattrain, y="Reaction Time (s)", x="Feature_Sensitivity_trainefc", ax=ax,
                    color=settings.lightteal_)
    ax.set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    ax.set_xlabel("Sensitivity for trained attention type")
    ax.set_ylabel("N-back RT for trained group")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    titlestring = "N-back RT Vs Motion Discrim Sensitivity"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # RT Correlations
    corrs_spacetrain = stats.pearsonr(i, j["Space_RT_trainefc"])
    corrs_feattrain = stats.pearsonr(k, l["Feature_RT_trainefc"])  # **********

    corrs_both = stats.pearsonr(pd.concat([i, k], axis=0),
                                pd.concat([j["Space_RT_trainefc"], l["Feature_RT_trainefc"]], axis=0))

    # df_correlationsdat_transfertasks.columns

    # Plot RT
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=NB_correlationsdat_Spacetrain, y="Reaction Time (s)", x="Space_RT_trainefc", ax=ax,
                    color=settings.yellow_)
    sns.scatterplot(data=NB_correlationsdat_Feattrain, y="Reaction Time (s)", x="Feature_RT_trainefc", ax=ax,
                    color=settings.lightteal_)
    ax.set_title("feature corr: r = " + str(corrs_feattrain[0]) + ', p = ' + str(corrs_feattrain[1]))
    ax.set_xlabel("RT for trained attention type")
    ax.set_ylabel("N-back RT for trained group")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    titlestring = "N-back RT Vs Motion Discrim RT"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Criterion Correlations
    corrs_spacetrain = stats.pearsonr(i, j["Space_Criterion_trainefc"])
    corrs_feattrain = stats.pearsonr(k, l["Feature_Criterion_trainefc"])  # **********

    corrs_both = stats.pearsonr(pd.concat([i, k], axis=0),
                                pd.concat([j["Space_Criterion_trainefc"], l["Feature_Criterion_trainefc"]], axis=0))

    # df_correlationsdat_transfertasks.columns

    # Plot RT
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=NB_correlationsdat_Spacetrain, y="Reaction Time (s)", x="Space_Criterion_trainefc", ax=ax,
                    color=settings.yellow_)
    sns.scatterplot(data=NB_correlationsdat_Feattrain, y="Reaction Time (s)", x="Feature_Criterion_trainefc", ax=ax,
                    color=settings.lightteal_)
    ax.set_title("feature corr: r = " + str(corrs_feattrain[0]) + ', p = ' + str(corrs_feattrain[1]))
    ax.set_xlabel("Criterion for trained attention type")
    ax.set_ylabel("N-back RT for trained group")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    titlestring = "N-back RT Vs Motion Discrim Criterion"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
