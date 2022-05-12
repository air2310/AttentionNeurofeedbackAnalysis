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

    amp_pre_space = list()
    amp_pre_feature = list()
    amp_post_space = list()
    amp_post_feature = list()
    amp_train_space = list()
    amp_train_feature = list()

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
            amplitude = np.mean(tmp, axis=0)

            select_pre_space.append(selectivity[0,0])
            select_pre_feature.append(selectivity[0, 1])
            select_post_space.append(selectivity[1, 0])
            select_post_feature.append(selectivity[1, 1])
            select_train_space.append(selectivity[1, 0] - selectivity[0, 0])
            select_train_feature.append(selectivity[1, 1] - selectivity[0, 1])

            amp_pre_space.append(amplitude[0,0])
            amp_pre_feature.append(amplitude[0, 1])
            amp_post_space.append(amplitude[1, 0])
            amp_post_feature.append(amplitude[1, 1])
            amp_train_space.append(amplitude[1, 0] - amplitude[0, 0])
            amp_train_feature.append(amplitude[1, 1] - amplitude[0, 1])

            traingroup_vec.append(attntrained)
            sub_vec.append(attntrained + str(sub_val))

    data = {'SubID': sub_vec, 'TrainingGroup': traingroup_vec, 'select_pre_space': select_pre_space,
            'select_pre_feature': select_pre_feature, 'select_post_space': select_post_space, 'select_post_feature': select_post_feature,
            'select_train_space': select_train_space, 'select_train_feature': select_train_feature, 'amp_pre_space': amp_pre_space,
            'amp_pre_feature': amp_pre_feature, 'amp_post_space': amp_post_space, 'amp_post_feature': amp_post_feature,
            'amp_train_space': amp_train_space, 'amp_train_feature': amp_train_feature}

    df_selectivity = pd.DataFrame(data)

    return df_selectivity


def load_SSVEPdata_electrodes(settings, day):

    # preallocate
    amp_space_attd = list()
    amp_feature_attd = list()
    amp_space_unattd = list()
    amp_feature_unattd = list()
    amp_space_select = list()
    amp_feature_select = list()

    electrode_vec = list()
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
            # results = np.load(bids.direct_results / Path(bids.substring + "EEG_pre_post_results_coherentmotionepochs.npz"), allow_pickle=True)
            # saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

            # store results
            # tmp = results['SSVEPs_prepost']
            tmp = results['SSVEPs_prepost_epochs'] #Settings.numchans, settings.num_attd_unattd, settings.num_days, settings.num_attnstates

            for i in np.arange(settings.num_electrodes):
                attd, state = 0, 0
                amp_space_attd.append(tmp[i, attd, day, state])

                attd, state = 0, 1
                amp_feature_attd.append(tmp[i, attd, day, state])

                attd, state = 1, 0
                amp_space_unattd.append(tmp[i, attd, day, state])

                attd, state = 1, 1
                amp_feature_unattd.append(tmp[i, attd, day, state])

                state = 0
                amp_space_select.append(tmp[i, 0, day, state] - tmp[i, 1, day, state])

                state = 1
                amp_feature_select.append(tmp[i, 0, day, state] - tmp[i, 1, day, state])

                electrode_vec.append(i)
                traingroup_vec.append(attntrained)
                sub_vec.append(attntrained + str(sub_val))


    data = {'SubID': sub_vec, 'TrainingGroup': traingroup_vec, 'electrode_vec': electrode_vec,
            'amp_space_attd': amp_space_attd, 'amp_space_unattd': amp_space_unattd,
            'amp_feature_attd': amp_feature_attd, 'amp_feature_unattd': amp_feature_unattd,
            'amp_space_select': amp_space_select, 'amp_feature_select': amp_feature_select}

    df_SSVEPS = pd.DataFrame(data)

    return df_SSVEPS


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

    # subjects to exlude
    behaveexcludestrings_space = np.array(['Feature1', 'Feature2', 'Feature23', 'Feature4', 'Feature9',
                                           'Sham10', 'Sham12', 'Sham16', 'Sham17', 'Sham3', 'Sham31',
                                           'Sham34', 'Sham39', 'Sham9', 'Space112', 'Space59', 'Space90'])
    behaveexcludestrings_feat = np.array(['Feature1', 'Feature2', 'Feature23', 'Feature4', 'Feature69',
                                          'Sham10', 'Sham16', 'Sham17', 'Sham22', 'Sham3', 'Sham31',
                                          'Sham39', 'Sham9', 'Space112', 'Space38', 'Space74', 'Space79',
                                          'Space90'])
    return df_sensitivity, bids, behaveexcludestrings_space, behaveexcludestrings_feat


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


def getcorrval_bytraininggroup(datuse, traininggroup, measureA, measureB):
    import scipy.stats as stats
    datusecorr = datuse.loc[datuse.TrainingGroup.isin(traininggroup),].copy()
    corr = stats.pearsonr(datusecorr[measureA], datusecorr[measureB])
    return np.round([element * 1000 for element in corr]) / 1000


def plotClasificationacc_correlation(datuse, measureA, measureB, settings, titlestring, ylims, excludeflag=False):
    df_sensitivity, bids, behaveexcludestrings_space, behaveexcludestrings_feat = load_MotionDiscrimBehaveResults(settings)

    fig, ax = plt.subplots(2, 3, figsize=(16, 10))
    colors = [settings.lightteal_, settings.darkteal_, settings.yellow_]
    NFstring = ["NF", "Sham"]
    for cue, cuestring in enumerate(measureB):
        # exclude the participants who couldn't do this cue
        if excludeflag:
            if cue == 0:
                datplot = datuse.drop(datuse[datuse['SubID'].isin(behaveexcludestrings_space)].index)
            if cue == 1:
                datplot = datuse.drop(datuse[datuse['SubID'].isin(behaveexcludestrings_feat)].index)
        else:
            datplot = datuse

        for training, trainingstring in enumerate(settings.string_attntrained):
            axuse = ax[cue, training]

            sns.scatterplot(data=datplot.loc[datplot.TrainingGroup.isin([trainingstring]),], x=measureA, y=cuestring, ax=axuse, color=colors[training])
            corr = getcorrval_bytraininggroup(datplot, traininggroup=[trainingstring], measureA=measureA, measureB=cuestring)
            print(corr)
            axuse.set_title(settings.string_attntype[cue] + " Cue, " + trainingstring + " Group, (r = " + str(corr[0]) + ', p = ' + str(corr[1]) + ")")
            axuse.set_ylim(ylims)
            axuse.set_xlim([50, 95])
            axuse.set_ylabel("Training effect")
            axuse.spines['top'].set_visible(False)
            axuse.spines['right'].set_visible(False)


    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


def plotscatter_bytraininggroups(datusecorrSp, datusecorrFt, titlestring, xlim, ylim, corr, bids, settings, measure):
    fig, axuse = plt.subplots(1, 1, figsize=(6, 6))

    sns.scatterplot(data=datusecorrSp, x='ClassifierAccuracy', y=measure, ax=axuse, color=settings.darkteal_)
    sns.scatterplot(data=datusecorrFt, x='ClassifierAccuracy', y=measure, ax=axuse, color=settings.lightteal_)
    axuse.legend(['Space', 'Feature'], title="Cue", loc='upper left')

    axuse.set_title("(r = " + str(corr[0]) + ', p = ' + str(corr[1]) + ")")

    axuse.set_ylim(ylim)
    axuse.set_xlim(xlim)
    axuse.spines['top'].set_visible(False)
    axuse.spines['right'].set_visible(False)

    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')


def plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt, datusecorrSh, titlestring, xlim, ylim, corr, corrsh, bids, settings, measure):
    fig, [axuse, axuse2] = plt.subplots(1, 2, figsize=(12, 6))

    sns.scatterplot(data=datusecorrSp, x=measure[0], y=measure[1], ax=axuse, color=settings.darkteal_)
    sns.scatterplot(data=datusecorrFt, x=measure[0], y=measure[1], ax=axuse, color=settings.lightteal_)
    axuse.legend(['Space', 'Feature'], title="Training Group", loc='upper right')
    axuse.set_title("NF: (r = " + str(corr[0]) + ', p = ' + str(corr[1]) + ")")
    axuse.set_ylim(ylim)
    axuse.set_xlim(xlim)
    axuse.spines['top'].set_visible(False)
    axuse.spines['right'].set_visible(False)

    sns.scatterplot(data=datusecorrSh, x=measure[0], y=measure[1], ax=axuse2, color=settings.yellow_)
    axuse2.legend(['Sham'], title="Training Group", loc='upper right')
    axuse2.set_title("Sham: (r = " + str(corrsh[0]) + ', p = ' + str(corrsh[1]) + ")")
    axuse2.set_ylim(ylim)
    axuse2.set_xlim(xlim)
    axuse2.spines['top'].set_visible(False)
    axuse2.spines['right'].set_visible(False)

    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


def ElectrodePerformanceRegression(settings):
    # TODO: Gather data from motion epochs (rather than full trial).
    import scipy.stats as stats
    from sklearn import linear_model
    import itertools
    from sklearn.model_selection import cross_validate

    # Get data
    df_sensitivity, bids, behaveexcludestrings_space, behaveexcludestrings_feat = load_MotionDiscrimBehaveResults(settings)

    # Cross validate Alpha
    ## Regression for space task
    predictors = ['amp_space_attd', 'amp_space_unattd', 'amp_space_select']

    num_iter = 126
    num_traininggroups = 3
    num_prepost = 2
    settings.num_electrodes = 9
    num_folds = 10
    alphas = [0.001, 0.005, 0.01, 0.7, 0.05, 0.2, 0.1, 0.5]
    RSquared_alpha = np.zeros((num_folds, len(alphas), num_traininggroups, num_prepost))
    RSquared_alpha[:] = np.nan

    for testday in np.arange(num_prepost):
        df_SSVEPs = load_SSVEPdata_electrodes(settings, day=testday)

        for alphacount, alpha in enumerate(alphas):
            for traininggroup, traininggroupstr in enumerate(settings.string_attntrained):
                y = df_sensitivity.loc[df_sensitivity.TrainingGroup == traininggroupstr,].reset_index()['sensitivity_pre_spacecue']
                datuse = df_SSVEPs.loc[df_SSVEPs.TrainingGroup == traininggroupstr,].reset_index()

                # choose all 9 electrodes
                electrodesuse = np.arange(settings.num_electrodes)
                dfuse = datuse.loc[datuse.electrode_vec.isin([0]),].reset_index()['SubID'].copy()  # get the data
                for chan in electrodesuse:
                    dfuse = pd.concat([dfuse, datuse.loc[datuse.electrode_vec.isin([chan]),].reset_index()[predictors]], axis=1)
                X = dfuse[predictors]

                # predict
                lm = linear_model.Lasso(alpha=alpha)  # linear_model.LinearRegression()
                lm.fit(X=X, y=y)
                cv_results = cross_validate(lm, X, y, cv=num_folds, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
                RSquared_alpha[:, alphacount, traininggroup, testday] = cv_results['test_r2']

    alpha_index=np.argmax(RSquared_alpha[:, :, :, :].mean(axis=0).mean(axis=1).mean(axis=1))
    print(alphas[alpha_index])
    alphause = 0.001 # 0.05

    ## Regression for space task
    predictors = ['amp_space_attd', 'amp_space_unattd', 'amp_space_select']
    num_iter = 126
    num_traininggroups = 3
    num_prepost = 2
    settings.num_electrodes = 9
    RSquared = np.zeros((num_folds, settings.num_electrodes, num_traininggroups, num_prepost, num_iter))
    RSquared[:] = np.nan


    for testday in np.arange(num_prepost):
        df_SSVEPs = load_SSVEPdata_electrodes(settings, day=testday)

        for traininggroup, traininggroupstr in enumerate(settings.string_attntrained):
            y = df_sensitivity.loc[df_sensitivity.TrainingGroup == traininggroupstr,].reset_index()['sensitivity_pre_spacecue']
            datuse = df_SSVEPs.loc[df_SSVEPs.TrainingGroup == traininggroupstr,].reset_index()

            for numchansuse in np.arange(1,settings.num_electrodes+1):

                electrodesuse_combos = list(itertools.combinations(np.arange(settings.num_electrodes), numchansuse))
                for iterations in np.arange(len(electrodesuse_combos)):
                    electrodesuse = electrodesuse_combos[0]  # choose a random group of electrodes to use

                    dfuse = datuse.loc[datuse.electrode_vec.isin([0]), ].reset_index()['SubID'].copy()  # get the data
                    for chan in electrodesuse:
                        dfuse = pd.concat([dfuse, datuse.loc[datuse.electrode_vec.isin([chan]), ].reset_index()[predictors]], axis=1)
                    X = dfuse[predictors]

                    # predict
                    # lm = linear_model.LinearRegression()
                    # lm.fit(X, y)
                    lm = linear_model.Lasso(alpha=0.01)  # linear_model.LinearRegression()
                    # lm.fit(X=X, y=y)
                    # RSquared[:, numchansuse-1, traininggroup, testday, iterations] = lm.score(X, y)
                    cv_results = cross_validate(lm, X, y, cv=num_folds, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
                    RSquared[:, numchansuse-1, traininggroup, testday, iterations]  = cv_results['test_r2']

    rplot  = np.nanmean(np.nanmean(RSquared, axis=0), axis=2)
    fig, axuse = plt.subplots(1, 3, figsize=(12, 6))
    axuse[0].plot(rplot[:, :, 0], '-x')
    axuse[0].legend(settings.string_attntrained)

    axuse[1].plot(rplot[:, :, 1], '-x')
    axuse[1].legend(settings.string_attntrained)

    dat =rplot[:, :, 1]- rplot[:, :, 0]
    axuse[2].plot(dat, '-x')
    axuse[2].legend(settings.string_attntrained)

    predictors = ['amp_feature_attd', 'amp_feature_unattd', 'amp_feature_select']
    num_iter = 126
    num_traininggroups = 3
    num_prepost = 2
    settings.num_electrodes = 9
    RSquared = np.zeros((num_folds, settings.num_electrodes, num_traininggroups, num_prepost, num_iter))
    RSquared[:] = np.nan

    for testday in np.arange(num_prepost):
        df_SSVEPs = load_SSVEPdata_electrodes(settings, day=testday)

        for traininggroup, traininggroupstr in enumerate(settings.string_attntrained):
            y = df_sensitivity.loc[df_sensitivity.TrainingGroup == traininggroupstr,].reset_index()['sensitivity_pre_featcue']
            datuse = df_SSVEPs.loc[df_SSVEPs.TrainingGroup == traininggroupstr,].reset_index()

            for numchansuse in np.arange(1, settings.num_electrodes + 1):

                electrodesuse_combos = list(itertools.combinations(np.arange(settings.num_electrodes), numchansuse))
                for iterations in np.arange(len(electrodesuse_combos)):
                    electrodesuse = electrodesuse_combos[0]  # choose a random group of electrodes to use

                    dfuse = datuse.loc[datuse.electrode_vec.isin([0]),].reset_index()['SubID'].copy()  # get the data
                    for chan in electrodesuse:
                        dfuse = pd.concat([dfuse, datuse.loc[datuse.electrode_vec.isin([chan]),].reset_index()[predictors]], axis=1)
                    X = dfuse[predictors]

                    # predict
                    # lm = linear_model.LinearRegression()
                    # lm.fit(X, y)
                    lm = linear_model.Lasso(alpha=alphause)  # linear_model.LinearRegression()
                    # lm.fit(X=X, y=y)
                    # RSquared[:, numchansuse-1, traininggroup, testday, iterations] = lm.score(X, y)
                    cv_results = cross_validate(lm, X, y, cv=num_folds, scoring=('r2', 'neg_mean_squared_error'), return_train_score=True)
                    RSquared[:, numchansuse - 1, traininggroup, testday, iterations] = cv_results['test_r2']

    rplot = np.nanmean(np.nanmean(RSquared, axis=0), axis=2)
    fig, axuse = plt.subplots(1, 3, figsize=(12, 6))
    axuse[0].plot(rplot[:, :, 0], '-x')
    axuse[0].legend(settings.string_attntrained)

    axuse[1].plot(rplot[:, :, 1], '-x')
    axuse[1].legend(settings.string_attntrained)

    dat = rplot[:, :, 1] - rplot[:, :, 0]
    axuse[2].plot(dat, '-x')
    axuse[2].legend(settings.string_attntrained)

    # next up - 9 electrodes but look at the weight changes
    ## Regression for space task
    predictors = ['amp_feature_attd', 'amp_feature_unattd', 'amp_feature_select']
    num_traininggroups = 3
    num_prepost = 2

    Weights = np.zeros((settings.num_electrodes*3, num_traininggroups, num_prepost))
    Weights[:] = np.nan

    for testday in np.arange(num_prepost):
        df_SSVEPs = load_SSVEPdata_electrodes(settings, day=testday)

        for traininggroup, traininggroupstr in enumerate(settings.string_attntrained):
            y = df_sensitivity.loc[df_sensitivity.TrainingGroup == traininggroupstr,].reset_index()['sensitivity_pre_featcue']
            datuse = df_SSVEPs.loc[df_SSVEPs.TrainingGroup == traininggroupstr,].reset_index()

            # choose all 9 electrodes
            electrodesuse = np.arange(settings.num_electrodes)
            dfuse = datuse.loc[datuse.electrode_vec.isin([0]),].reset_index()['SubID'].copy()  # get the data
            for chan in electrodesuse:
                dfuse = pd.concat([dfuse, datuse.loc[datuse.electrode_vec.isin([chan]),].reset_index()[predictors]], axis=1)
            X = dfuse[predictors]

            # predict
            lm = linear_model.Lasso(alpha=0.00001) #linear_model.LinearRegression()
            lm.fit(X=X, y=y)
            Weights[:, traininggroup, testday] = lm.coef_
            print(lm.score(X, y))

    fig, axuse = plt.subplots(1, 3, figsize=(12, 6))
    cond =2
    axuse[0].bar(np.arange(9*3), Weights[:, cond, 0])
    # axuse[0].set_ylim([-50,50])

    axuse[1].bar(np.arange(9*3), Weights[:, cond, 1])
    # axuse[1].set_ylim([-50,50])

    dat = Weights[:, cond, 1] - Weights[:, cond, 0]
    axuse[2].bar(np.arange(9*3), dat)
    # axuse[2].set_ylim([-50,50])


def classification_acc_correlations(settings):
    import scipy.stats as stats

    # Get data
    df_classifier, df_classifier_condensed = load_classifierdata(settings)
    df_selectivity = load_SSVEPdata(settings)
    df_sensitivity, bids, behaveexcludestrings_space, behaveexcludestrings_feat = load_MotionDiscrimBehaveResults(settings)

    # sensitivity pre vs post
    fig, [axuse, axuse2] = plt.subplots(1, 2, figsize=(12, 6))

    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup']=="Space"], x='sensitivity_pre_spacecue', y='sensitivity_post_spacecue', ax=axuse, color=settings.darkteal_)
    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup']=="Feature"], x='sensitivity_pre_spacecue', y='sensitivity_post_spacecue', ax=axuse, color=settings.lightteal_)
    axuse.legend(['Space', 'Feature'], title="Training Group", loc='upper left')

    axuse.set_ylim([-1, 4])
    axuse.set_xlim([-1, 4])
    axuse.spines['top'].set_visible(True)
    axuse.spines['right'].set_visible(True)

    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup']=="Sham"], x='sensitivity_pre_spacecue', y='sensitivity_post_spacecue', ax=axuse2, color=settings.yellow_)
    axuse2.legend(['Sham'], title="Training Group", loc='upper left')

    axuse2.set_ylim([-1, 4])
    axuse2.set_xlim([-1, 4])
    axuse2.spines['top'].set_visible(False)
    axuse2.spines['right'].set_visible(False)

    titlestring ='prepost consistency'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # sensitivity space vs feat
    df_sensitivity['sensitivity_spacecue'] = (df_sensitivity['sensitivity_pre_spacecue'] + df_sensitivity['sensitivity_post_spacecue']) / 2
    df_sensitivity['sensitivity_featcue'] = (df_sensitivity['sensitivity_pre_featcue'] + df_sensitivity['sensitivity_post_featcue']) / 2
    fig, axuse = plt.subplots(2, 3, figsize=(12, 10))

    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup'] == "Space"], x='sensitivity_spacecue', y='sensitivity_featcue', ax=axuse[0,0], color=settings.darkteal_)
    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup'] == "Feature"], x='sensitivity_spacecue', y='sensitivity_featcue', ax=axuse[0,1], color=settings.lightteal_)
    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup'] == "Sham"], x='sensitivity_spacecue', y='sensitivity_featcue', ax=axuse[0,2], color=settings.yellow_)

    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup'] == "Space"], x='sensitivity_train_spacecue', y='sensitivity_train_featcue', ax=axuse[1, 0], color=settings.darkteal_)
    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup'] == "Feature"], x='sensitivity_train_spacecue', y='sensitivity_train_featcue', ax=axuse[1, 1], color=settings.lightteal_)
    sns.scatterplot(data=df_sensitivity[df_sensitivity['TrainingGroup'] == "Sham"], x='sensitivity_train_spacecue', y='sensitivity_train_featcue', ax=axuse[1, 2], color=settings.yellow_)
    for i in np.arange(3):
        axuse[0, i].set_xlim([-1, 4])
        axuse[0, i].set_ylim([-1, 2])
        axuse[0, i].spines['top'].set_visible(False)
        axuse[0, i].spines['right'].set_visible(False)

        axuse[1, i].set_xlim([-1, 2])
        axuse[1, i].set_ylim([-1.5, 1.5])
        axuse[1, i].spines['top'].set_visible(False)
        axuse[1, i].spines['right'].set_visible(False)

        corr = stats.pearsonr(df_sensitivity[df_sensitivity['TrainingGroup'] == settings.string_attntrained[i]]['sensitivity_train_spacecue'], df_sensitivity[df_sensitivity['TrainingGroup'] == settings.string_attntrained[i]]['sensitivity_train_featcue'])

    corr = stats.pearsonr(df_sensitivity['sensitivity_train_featcue'], df_sensitivity['sensitivity_train_spacecue'])
    print(corr) #0.4686849006858813, 3.125935127100983e-07

    titlestring = 'correlations feat vs space sensitivity'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    # plot classifier accuracy by attention type
    plot_classificationacc(df_classifier_condensed, bids, settings)



    # Correlations!
    # Classification Acc V Sensitivity training effect
    datuse = pd.concat([df_classifier_condensed[['SubID', 'TrainingGroup', 'ClassifierAccuracy']], df_sensitivity[['sensitivity_train_spacecue', 'sensitivity_train_featcue']]], axis=1)
    plotClasificationacc_correlation(datuse=datuse, measureA='ClassifierAccuracy',measureB=['sensitivity_train_spacecue', 'sensitivity_train_featcue'], settings=settings, titlestring= "Classifier Acc Vs. Sensitivity Scatter", ylims = [-2,2], excludeflag = False)




    # Plot average training effect correlation
    df_sensitivity['sensitivity_traineffect'] = df_sensitivity[['sensitivity_train_spacecue', 'sensitivity_train_featcue']].mean(axis=1)

    df_sensitivity['sensitivity_precue'] = df_sensitivity[['sensitivity_pre_spacecue', 'sensitivity_pre_featcue']].min(axis=1)
    tmp1 = pd.merge(df_classifier_condensed[['SubID', 'TrainingGroup', 'ClassifierAccuracy']], df_sensitivity)

    # spaceexclude = ['Space45', 'Space46', 'Space52', 'Space54', 'Space60', 'Space64', 'Space97', 'Space99', 'Space128', 'Feature18', 'Feature58', 'Feature72', 'Feature76', 'Feature87', 'Feature89', 'Feature117', 'Feature119', 'Feature120', 'Sham4', 'Sham7', 'Sham19', 'Sham21', 'Sham23', 'Sham25', 'Sham26', 'Sham30', 'Sham35']
    # # featureexclude =['Space29', 'Space52', 'Space54', 'Space60', 'Space64', 'Space99', 'Space107', 'Space125', 'Space128', 'Feature8', 'Feature57', 'Feature58', 'Feature63', 'Feature72', 'Feature87', 'Feature117', 'Feature119', 'Feature120', 'Sham2', 'Sham4', 'Sham18', 'Sham19', 'Sham21', 'Sham23', 'Sham26', 'Sham30', 'Sham35']
    # tmp1 = tmp1[(~tmp1.SubID.isin(spaceexclude)) ]#& (~tmp1.SubID.isin(featureexclude)) ]
    # tmp = tmp1
    tmp1 = tmp1[tmp1['sensitivity_precue']>0.1]
    tmp = tmp1[tmp1['ClassifierAccuracy']<60]

    fig, ax = plt.subplots(1,1, figsize=(5,5))
    trainingstring = [ "Space", "Feature"]  #["Sham"] #
    sns.scatterplot(data=tmp.loc[tmp.TrainingGroup.isin(trainingstring),], x='ClassifierAccuracy', y='sensitivity_traineffect', ax=ax, color=settings.darkteal_)
    corr = stats.pearsonr(tmp.loc[tmp.TrainingGroup.isin(trainingstring),'ClassifierAccuracy'], tmp.loc[tmp.TrainingGroup.isin(trainingstring),'sensitivity_traineffect'])
    print(corr)

    datuse = tmp.loc[tmp.TrainingGroup.isin(trainingstring),]
    datuse["median_split"] = (datuse.ClassifierAccuracy<datuse.ClassifierAccuracy.quantile()).replace({True:1, False:2})
    datuse.groupby("median_split").mean()
    stats.ttest_1samp(datuse[datuse["median_split"] == 1]["sensitivity_traineffect"],0)
    stats.ttest_1samp(datuse[datuse["median_split"] == 2]["sensitivity_traineffect"],0)


    # Classification Acc V Selectivity training effect
    # datuse = pd.concat([df_classifier_condensed['SubID'], df_classifier_condensed['TrainingGroup'], df_classifier_condensed['ClassifierAccuracy'], abs(df_selectivity['select_train_space']), abs(df_selectivity['select_train_feature'])], axis=1)
    datuse = pd.concat([df_classifier_condensed[['SubID', 'TrainingGroup', 'ClassifierAccuracy']], df_selectivity[['select_train_space','select_train_feature']]], axis=1)
    plotClasificationacc_correlation(datuse=datuse, measureA='ClassifierAccuracy', measureB=['select_train_space', 'select_train_feature'], settings=settings, titlestring="Classifier Acc Vs. Selectivity Scatter", ylims = [-0.5, 0.5])


    # Classification Acc V SSVEP Amp training effect
    datuse = pd.concat([df_classifier_condensed[['SubID', 'TrainingGroup', 'ClassifierAccuracy']], df_selectivity[['amp_train_space', 'amp_train_feature']]], axis=1)
    plotClasificationacc_correlation(datuse=datuse, measureA='ClassifierAccuracy', measureB=['amp_train_space', 'amp_train_feature'], settings=settings, titlestring="Classifier Acc Vs. SSVEP Amp Scatter", ylims=[-0.5, 0.5])


    ## Classification Acc V selectivity (day 1) - we'll get the non-condensed classifier data for this so we can look at what's going on with the classifier not used to generate NF
    df_classifier_space = df_classifier.loc[df_classifier['ClassifierType'] == 'Space',].reset_index()
    df_classifier_feature = df_classifier.loc[df_classifier['ClassifierType'] == 'Feature',].reset_index()

    df_classifier_space = pd.concat([df_classifier_space, df_selectivity[['select_pre_space', 'amp_pre_space']]], axis=1).reset_index().rename(columns={'select_pre_space': 'select_pre', 'amp_pre_space': 'amp_pre'})
    df_classifier_feature = pd.concat([df_classifier_feature, df_selectivity[['select_pre_feature', 'amp_pre_feature']]], axis=1).reset_index().rename(columns={'select_pre_feature': 'select_pre', 'amp_pre_feature': 'amp_pre'})

    df_classifier_all = pd.concat([df_classifier_space, df_classifier_feature], axis=0)
    df_classifier_all = df_classifier_all.loc[df_classifier_all.TrainingGroup.isin(["Space", "Feature"]),]

    datusecorrSp = df_classifier_all.loc[df_classifier_all.ClassifierType.isin(["Space"])   & df_classifier_all.AttentionTrained.isin(["Space"  ]), ].copy()
    datusecorrFt = df_classifier_all.loc[df_classifier_all.ClassifierType.isin(["Feature"]) & df_classifier_all.AttentionTrained.isin(["Feature"]), ].copy()
    datusecorrAll = pd.concat([datusecorrSp, datusecorrFt])
    corr = stats.pearsonr(datusecorrAll['ClassifierAccuracy'], datusecorrAll['select_pre'])
    plotscatter_bytraininggroups(datusecorrSp, datusecorrFt, titlestring = "Classification Accuracy Vs SSVEP Selectivity", xlim=[50, 95], ylim = [-0.25, 0.8], corr = corr, bids = bids, settings = settings, measure = 'select_pre')

    corr = stats.pearsonr(datusecorrAll['ClassifierAccuracy'], datusecorrAll['amp_pre'])
    plotscatter_bytraininggroups(datusecorrSp, datusecorrFt, titlestring="Classification Accuracy Vs SSVEP Amp", xlim=[50, 95], ylim=[-0.25, 0.8], corr=corr, bids=bids, settings=settings, measure='amp_pre')

    datusecorrSp = df_classifier_all.loc[df_classifier_all.ClassifierType.isin(["Space"]),].copy()
    datusecorrFt = df_classifier_all.loc[df_classifier_all.ClassifierType.isin(["Feature"]),].copy()
    datusecorrAll = pd.concat([datusecorrSp, datusecorrFt])
    corr = stats.pearsonr(datusecorrAll['ClassifierAccuracy'], datusecorrAll['select_pre'])
    plotscatter_bytraininggroups(datusecorrSp, datusecorrFt, titlestring="Classification Accuracy Vs SSVEP Selectivity All Participants", xlim=[40, 95], ylim=[-0.25, 1.2], corr=corr, bids = bids, settings = settings, measure = 'select_pre')


    ## Classification Acc V sensitivity (day 1)
    tmp = pd.concat([df_classifier_condensed['ClassifierAccuracy'], df_sensitivity], axis=1)
    tmp = tmp.loc[tmp.TrainingGroup.isin(["Space", "Feature"]), ].copy()

    datusecorrSp = tmp.loc[tmp.TrainingGroup.isin(["Space"]), ].copy()
    datusecorrFt = tmp.loc[tmp.TrainingGroup.isin(["Feature"]), ].copy()
    datusecorrAll = tmp.copy()
    corr = stats.pearsonr(datusecorrSp['ClassifierAccuracy'], datusecorrSp['sensitivity_pre_spacecue'])
    plotscatter_bytraininggroups(datusecorrSp, datusecorrFt, measure = 'sensitivity_pre_spacecue', titlestring = "Classification Accuracy Vs Sensitivity pre training", xlim=[50, 95], ylim = [-1, 4], corr = corr, bids = bids, settings = settings)


    ## Selectivity V sensitivity
    tmp = pd.concat([df_selectivity[['select_pre_space', 'select_post_space', 'select_pre_feature', 'select_post_feature', 'select_train_space', 'select_train_feature']], df_sensitivity], axis=1)
    # tmp = tmp.loc[tmp.TrainingGroup.isin(["Space", "Feature"]),].copy()

    datusecorrSp = tmp.loc[tmp.TrainingGroup.isin(["Space"]),].copy()
    datusecorrFt = tmp.loc[tmp.TrainingGroup.isin(["Feature"]),].copy()
    datusecorrSh = tmp.loc[tmp.TrainingGroup.isin(["Sham"]),].copy()
    datusecorrAll = tmp.copy()
    datusecorrSpFt = pd.concat([datusecorrSp, datusecorrFt]).copy()

    corr = stats.pearsonr(datusecorrSpFt['select_post_space'], datusecorrSpFt['sensitivity_post_spacecue'])
    corr = stats.pearsonr(datusecorrSh['select_post_space'], datusecorrSh['sensitivity_post_spacecue'])
    corr = stats.pearsonr(datusecorrFt['select_post_space'], datusecorrFt['sensitivity_post_spacecue'])

    corr = stats.pearsonr(datusecorrSpFt['select_pre_space'], datusecorrSpFt['sensitivity_pre_spacecue'])
    corrsh = stats.pearsonr(datusecorrSh['select_pre_space'], datusecorrSh['sensitivity_pre_spacecue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt, datusecorrSh, measure=['select_pre_space', 'sensitivity_pre_spacecue'], titlestring="Selectivity Vs Sensitivity pre training space", xlim=[-0.25, 1], ylim=[-0.5, 4], corr=corr, corrsh=corrsh, bids=bids, settings=settings)

    corr = stats.pearsonr(datusecorrSpFt['select_pre_feature'], datusecorrSpFt['sensitivity_pre_featcue'])
    corrsh = stats.pearsonr(datusecorrSh['select_pre_feature'], datusecorrSh['sensitivity_pre_featcue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt,datusecorrSh, measure=['select_pre_feature', 'sensitivity_pre_featcue'], titlestring="Selectivity Vs Sensitivity pre training feat", xlim=[-0.1, 0.4], ylim=[-0.5, 2], corr=corr, corrsh=corrsh, bids=bids, settings=settings)

    corr = stats.pearsonr(datusecorrSpFt['select_post_space'], datusecorrSpFt['sensitivity_post_spacecue'])
    corrsh = stats.pearsonr(datusecorrSh['select_post_space'], datusecorrSh['sensitivity_post_spacecue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt,datusecorrSh, measure=['select_post_space', 'sensitivity_post_spacecue'], titlestring="Selectivity Vs Sensitivity post training space", xlim=[-0.2, 1.0], ylim=[-0.75, 4], corr=corr, corrsh=corrsh, bids=bids, settings=settings)

    corr = stats.pearsonr(datusecorrSpFt['select_post_feature'], datusecorrSpFt['sensitivity_post_featcue'])
    corrsh = stats.pearsonr(datusecorrSh['select_post_feature'], datusecorrSh['sensitivity_post_featcue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt, datusecorrSh, measure=['select_post_feature', 'sensitivity_post_featcue'], titlestring="Selectivity Vs Sensitivity post training feat", xlim=[-0.1, 0.3], ylim=[-0.75, 3], corr=corr, corrsh=corrsh, bids=bids, settings=settings)


    corr = stats.pearsonr(datusecorrSpFt['select_train_space'], datusecorrSpFt['sensitivity_pre_spacecue'])
    corrsh = stats.pearsonr(datusecorrSh['select_train_space'], datusecorrSh['sensitivity_pre_spacecue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt,datusecorrSh, measure=['select_train_space', 'sensitivity_pre_spacecue'], titlestring="Selectivity train Vs Sensitivity pre training space", xlim=[-0.3, 0.6], ylim=[-0.75, 4], corr=corr, corrsh=corrsh, bids=bids, settings=settings)


    corr = stats.pearsonr(datusecorrSpFt['select_train_space'], datusecorrSpFt['sensitivity_post_spacecue'])
    corrsh = stats.pearsonr(datusecorrSh['select_train_space'], datusecorrSh['sensitivity_post_spacecue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt,datusecorrSh, measure=['select_train_space', 'sensitivity_post_spacecue'], titlestring="Selectivity train Vs Sensitivity post training space", xlim=[-0.3, 0.6], ylim=[-0.75, 4], corr=corr, corrsh=corrsh, bids=bids, settings=settings)


    corr = stats.pearsonr(datusecorrSpFt['select_train_space'], datusecorrSpFt['sensitivity_train_spacecue'])
    corrsh = stats.pearsonr(datusecorrSh['select_train_space'], datusecorrSh['sensitivity_train_spacecue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt,datusecorrSh, measure=['select_train_space', 'sensitivity_train_spacecue'], titlestring="Selectivity train Vs Sensitivity train space", xlim=[-0.6, 0.6], ylim=[-1, 2.5], corr=corr, corrsh=corrsh, bids=bids, settings=settings)


    corr = stats.pearsonr(datusecorrSpFt['select_train_feature'], datusecorrSpFt['sensitivity_train_featcue'])
    corrsh = stats.pearsonr(datusecorrSh['select_train_feature'], datusecorrSh['sensitivity_train_featcue'])
    plotscatter_bytraininggroups_sham(datusecorrSp, datusecorrFt,datusecorrSh, measure=['select_train_feature', 'sensitivity_train_featcue'], titlestring="Selectivity train Vs Sensitivity train feat", xlim=[-0.6, 0.6], ylim=[-1, 2.5], corr=corr, corrsh=corrsh, bids=bids, settings=settings)

