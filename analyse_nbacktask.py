import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import helperfunctions_ATTNNF as helper


def analyse_nbacktask(settings, sub_val):
    # get task specific settings
    settings = settings.get_settings_nbacktask()

    # pre-allocate
    acc_nback = np.empty((settings.num_trials, settings.num_days))
    rt_nback = np.empty((settings.num_trials, settings.num_days))

    # loop through days
    for day_count, day_val in enumerate(settings.daysuse):
        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)

        # decide which file to use
        possiblefiles = []
        filesizes = []
        for filesfound in bids.direct_data_behave.glob(bids.filename_nback + "*.mat"):
            filesizes.append(filesfound.stat().st_size)
            possiblefiles.append(filesfound)

        file2useIDX = np.argmax(
            filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
        file2use = possiblefiles[file2useIDX]

        # load data
        F = h5py.File(file2use, 'r')
        print(list(F.keys()))

        # get Accuracy
        acc_nback[:, day_count] = np.array(F['ACC_ALL']).reshape(
            settings.num_trials)  # responses.hit = 1;responses.falsealarm = 2;responses.correctreject = 3; responses.miss = 4;
        rt_nback[:, day_count] = np.array(F['RT_ALL']).reshape(settings.num_trials)

    ############## Get Sensitivity

    N_distractors = 132  # number of distractor events per day and condition.
    N_targets = 60  # number of target events per day and condition.

    HIT = np.array([np.sum(acc_nback[:, 0] == 1)/N_targets, np.sum(acc_nback[:, 1] == 1)/N_targets])  # hitrate
    FA = np.array([np.sum(acc_nback[:, 0] == 2) / N_distractors, np.sum(acc_nback[:, 1] == 2) / N_distractors])  # FArate


    # Plot N-back pre vs. post Results

    import seaborn as sns
    import pandas as pd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.set(style="ticks")
    colors = ["#112F41", "#4CB99F"]
    # Accuracy

    data = {'Testday': settings.string_prepost, 'Hits': HIT*100, 'FAs': FA*100}
    df = pd.DataFrame(data)

    # Grouped violinplot
    sns.barplot(x="Testday", y="Hits", data=df, palette=sns.color_palette(colors), ax=ax1)
    sns.barplot(x="Testday", y="FAs", data=df, palette=sns.color_palette([settings.orange, settings.yellow]), ax=ax1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_ylabel('Rate (%)')

    ax1.set_title("Hits and FAs")

    # Reaction time
    data = {'Testday': [settings.string_prepost[0]] * settings.num_trials + [
        settings.string_prepost[1]] * settings.num_trials,
            'RT': np.concatenate((rt_nback[:, 0], rt_nback[:, 1])),
            'ACC': np.concatenate((acc_nback[:, 0], acc_nback[:, 1]))
            }

    meanrt = np.array([np.mean(rt_nback[np.isin(acc_nback[:, 0], [1, 3]), 0]), np.mean(rt_nback[np.isin(acc_nback[:, 1], [1, 3]), 1])])
    df = pd.DataFrame(data)

    # Grouped violinplot
    # responses.hit = 1;responses.falsealarm = 2;responses.correctreject = 3; responses.miss = 4;
    sns.violinplot(x="Testday", y="RT", hue="ACC", data=df, palette=sns.color_palette([settings.lightteal, settings.red, settings.medteal, settings.orange]), style="ticks", ax=ax2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles, ['hit', 'falsealarm', 'correctreject', 'miss'])

    ax2.set_ylabel('Reaction Time (s)')
    ax2.set_title("Reaction time")

    titlestring = bids.substring + ' Nback Results train ' + settings.string_attntrained[
        settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # save results
    np.savez(bids.direct_results / Path(bids.substring + "Nback_results"),
             hits=HIT, falsealarms=FA, meanrt=meanrt, acc_nback=acc_nback, rt_nback=rt_nback)


def collate_nbacktask(settings):
    import seaborn as sns
    import pandas as pd

    # preallocate
    num_subs = np.zeros((settings.num_attnstates+1))
    substring = []
    daystrings = []
    attnstrings = []
    accuracy_compare = []
    rt_compare = []

    print('Collating N-back Task for space and feature train')

    for attntrained in np.arange(settings.num_attnstates+1):  # cycle trough space and feature train groups

        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_nbacktask()

        # pre-allocate for this group
        if (attntrained == 0):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([90])))
            settings.num_subs = settings.num_subs - 1

        if (attntrained == 1):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([89])))
            settings.num_subs = settings.num_subs - 1

        if (attntrained == 2):  # correct for lost data for space train
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([15])))
            settings.num_subs = settings.num_subs - 1

        num_subs[attntrained] = settings.num_subs
        mean_acc_all = np.empty((settings.num_days, settings.num_subs))
        mean_rt_all = np.empty((settings.num_days, settings.num_subs))
        substring_short = []

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "Nback_results.npz"),
                              allow_pickle=True)  # saved vars: meanacc=meanacc, meanrt=meanrt, acc_nback=acc_nback, rt_nback=rt_nback

            # store results temporarily
            mean_acc_all[:, sub_count] = results['meanacc'] * 100
            mean_rt_all[:, sub_count] = results['meanrt']

            substring_short = np.concatenate((substring_short, [bids.substring]))

        # store results for attention condition
        tmp = np.concatenate((substring_short, substring_short))
        substring = np.concatenate((substring, tmp))

        # store results for attention condition
        tmp = [settings.string_prepost[0]] * settings.num_subs + [settings.string_prepost[1]] * settings.num_subs
        daystrings = np.concatenate((daystrings, tmp))

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days
        attnstrings = np.concatenate((attnstrings, tmp))

        tmp = np.concatenate((mean_acc_all[0, :], mean_acc_all[1, :]))
        accuracy_compare = np.concatenate((accuracy_compare, tmp))

        tmp = np.concatenate((mean_rt_all[0, :], mean_rt_all[1, :]))
        rt_compare = np.concatenate((rt_compare, tmp))

    # create the data frames for accuracy and reaction time data
    data = {'SubID': substring, 'Testday': daystrings, 'Attention Trained': attnstrings,
            'Accuracy (%)': accuracy_compare, 'Reaction Time (s)': rt_compare}
    df_acc = pd.DataFrame(data)

    # data = {'SubID': substring, 'Testday': daystrings, 'Attention Trained': attnstrings, 'Reaction Time (s)': rt_compare}
    # df_rt = pd.DataFrame(data)

    # Get effects
    means=df_acc.groupby(['Attention Trained', 'SubID' ]).mean()

    df_acc_effects = df_acc.copy()

    for idx, str in enumerate(settings.string_attntrained):
        for idx2, testday in enumerate(settings.string_prepost):
            for SS in range(len(means['Accuracy (%)'][str])):
                # get the trials we want to affect
                trialsuse = df_acc['Attention Trained'].isin([str]) & df_acc['Testday'].isin([testday])
                tmp = df_acc[trialsuse]['SubID'].reset_index()['SubID'][SS];
                trialsuse2 = trialsuse & df_acc['SubID'].isin([tmp])

                # get correct mean
                tmpM = means['Accuracy (%)'][str].index == tmp

                # extract data
                dat = df_acc[trialsuse2]['Accuracy (%)'].reset_index()
                df_acc_effects.loc[trialsuse2, 'Accuracy (%)'] = dat['Accuracy (%)'][0] - means['Accuracy (%)'][str][tmpM][0]

                dat = df_acc[trialsuse2]['Reaction Time (s)'].reset_index()
                df_acc_effects.loc[trialsuse2, 'Reaction Time (s)'] = dat['Reaction Time (s)'][0] - means['Reaction Time (s)'][str][tmpM][0]

    # plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = [settings.lightteal, settings.medteal]

    # Accuracy Grouped violinplot

    sns.swarmplot(x="Attention Trained", y="Accuracy (%)", hue="Testday", dodge=True, data=df_acc, ax=ax1, color="0", alpha=0.3)
    sns.violinplot(x="Attention Trained", y="Accuracy (%)", hue="Testday", data=df_acc,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=False)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], labels[:2])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim(0, 100)
    ax1.set_title("Accuracy")

    # Reaction time Grouped violinplot
    colors = [settings.lightteal, settings.medteal]
    sns.swarmplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", dodge=True, data=df_acc, ax=ax2, color="0", alpha=0.3)
    sns.violinplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", data=df_acc,
                   palette=sns.color_palette(colors), style="ticks", ax=ax2, split=False)
    handles, labels = ax2.get_legend_handles_labels()
    ax2.legend(handles[:2], labels[:2])

    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title("Reaction time")

    titlestring = 'Nback Results Compare Training sub mean subtracted'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # save out
    df_acc.to_pickle(bids.direct_results_group / Path("group_Nback.pkl"))