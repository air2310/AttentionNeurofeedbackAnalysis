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
    hits_compare = []
    FAs_compare = []
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
        mean_hits_all = np.empty((settings.num_days, settings.num_subs))
        mean_fas_all = np.empty((settings.num_days, settings.num_subs))
        mean_rt_all = np.empty((settings.num_days, settings.num_subs))
        substring_short = []

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "Nback_results.npz"),
                              allow_pickle=True)  # saved vars:  hits=HIT, falsealarms=FA, meanrt=meanrt, acc_nback=acc_nback, rt_nback=rt_nback)

            # store results temporarily
            mean_hits_all[:, sub_count] = results['hits'] * 100
            mean_fas_all[:, sub_count] = results['falsealarms'] *100
            mean_rt_all[:, sub_count] = results['meanrt']

            substring_short = np.concatenate((substring_short, [sub_count + attntrained*37]))

        # store results for attention condition
        tmp = np.concatenate((substring_short, substring_short))
        substring = np.concatenate((substring, tmp))

        # store results for attention condition
        tmp = [settings.string_prepost[0]] * settings.num_subs + [settings.string_prepost[1]] * settings.num_subs
        daystrings = np.concatenate((daystrings, tmp))

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days
        attnstrings = np.concatenate((attnstrings, tmp))

        tmp = np.concatenate((mean_hits_all[0, :], mean_hits_all[1, :]))
        hits_compare = np.concatenate((hits_compare, tmp))

        tmp = np.concatenate((mean_fas_all[0, :], mean_fas_all[1, :]))
        FAs_compare = np.concatenate((FAs_compare, tmp))

        tmp = np.concatenate((mean_rt_all[0, :], mean_rt_all[1, :]))
        rt_compare = np.concatenate((rt_compare, tmp))

    # create the data frames for accuracy and reaction time data
    data = {'SubID': substring, 'Testday': daystrings, 'Attention Trained': attnstrings,
            'HITS (%)': hits_compare, 'False Alarms (%)': FAs_compare, 'Reaction Time (s)': rt_compare}
    df_acc = pd.DataFrame(data)

    # data = {'SubID': substring, 'Testday': daystrings, 'Attention Trained': attnstrings, 'Reaction Time (s)': rt_compare}
    # df_rt = pd.DataFrame(data)

    ############## Get Sensitivity
    from scipy.stats import norm

    Z = norm.ppf  # percentile point function - normal distribution between 0 and 1.

    N_distractors = 132  # number of distractor events per day and condition.
    N_targets = 60  # number of target events per day and condition.

    dat = df_acc.loc[:, "HITS (%)"] / 100  # hitrate

    dat[dat == 0] = 1 / (2 * N_targets)  # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_targets)  # correct for zeros and ones (McMillan & Creelman, 2004)

    hitrate_zscore = Z(dat)

    dat = df_acc.loc[:, "False Alarms (%)"] / 100  # False Alarm rate

    dat[dat == 0] = 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)

    falsealarmrate_zscore = Z(dat)

    df_acc.loc[:, "Sensitivity"] = hitrate_zscore - falsealarmrate_zscore
    df_acc.loc[:, "Criterion"] = 0.5 * (hitrate_zscore + falsealarmrate_zscore)
    df_acc.loc[:, "LikelihoodRatio"] =df_acc.loc[:, "Sensitivity"] * df_acc.loc[:, "Criterion"]

    # clear up really bad performers
    reject = df_acc.loc[df_acc["Sensitivity"]<0,:]
    df_acc = df_acc.loc[~df_acc["SubID"].isin(reject["SubID"]),:]
    #
    # df_acc = df_acc[~df_acc['SubID'].isin([3, 4, 6, 7, 10, 12, 14, 18, 19, 23, 24, 27, 28,
    #                                                   29, 37, 38, 39, 40, 41, 43, 44, 51, 52, 58, 64, 74,
    #                                                   76, 78, 79, 81, 82, 83, 85, 86, 88, 89, 90, 95, 100,
    #                                                   102, 104, 107, 110])]

    print(len(df_acc[df_acc['Attention Trained'] == 'Sham'].SubID.unique()))

    # plot results
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 15))
    sns.set(style="ticks")
    colors = [settings.lightteal, settings.medteal]

    # Accuracy Grouped violinplot

    sns.swarmplot(x="Attention Trained", y="Sensitivity", hue="Testday", dodge=True, data=df_acc, ax=ax1, color="0", alpha=0.3)
    sns.violinplot(x="Attention Trained", y="Sensitivity", hue="Testday", data=df_acc,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=False)

    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], labels[:2])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim(0, 100)
    ax1.set_title("Sensitivity")

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

    titlestring = 'Nback Results Compare Training'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # Save results for processing in R
    df_acc.to_csv(bids.direct_results_group_compare / Path("NBACK_behaveresults_ALL.csv"), index=False)
    df_acc.to_pickle(bids.direct_results_group / Path("group_Nback.pkl"))

    # N-back training effects
    idx_d1 = df_acc["Testday"] == "pre-training"
    idx_d4 = df_acc["Testday"]  == "post-training"

    tmpd4 = df_acc[idx_d4].reset_index()
    tmpd1 = df_acc[idx_d1].reset_index()

    df_behtraineffects = tmpd4[["SubID", "Attention Trained"]].copy()

    df_behtraineffects["∆Sensitivity"] = tmpd4['Sensitivity'] - tmpd1['Sensitivity']
    df_behtraineffects["∆Criterion"] = tmpd4['Criterion'] - tmpd1['Criterion']
    df_behtraineffects["∆HITS"] = tmpd4['HITS (%)'] - tmpd1['HITS (%)']
    df_behtraineffects["∆Reaction Time (s)"] = tmpd4['Reaction Time (s)'] - tmpd1['Reaction Time (s)']

    # Plot sensitivity training effect
    colors = [settings.yellow, settings.orange, settings.red]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))

    measurestrings = ["∆Sensitivity", "∆Reaction Time (s)"]
    for i in np.arange(2):
        sns.swarmplot(x="Attention Trained", y=measurestrings[i], data=df_behtraineffects, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Trained", y=measurestrings[i], data=df_behtraineffects, palette=sns.color_palette(colors), style="ticks",
                       ax=ax[i], inner="box", alpha=0.6)
        sns.lineplot(x="Attention Trained", y=measurestrings[i], data=df_behtraineffects, markers=True, dashes=False, color="k", err_style="bars", ci=68, ax=ax[i])

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(measurestrings[i])

    ax[i].set_ylim([-.4, .2])
    titlestring = "Nback task training effects"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.eps'), format='eps')
