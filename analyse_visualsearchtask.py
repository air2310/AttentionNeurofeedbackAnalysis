import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import helperfunctions_ATTNNF as helper


def analyse_visualsearchtask(settings, sub_val):
    # get task specific settings
    settings = settings.get_settings_visualsearchtask()

    # pre-allocate
    acc_vissearch = np.empty((settings.num_trialscond, settings.num_setsizes, settings.num_days))
    rt_vissearch = np.empty((settings.num_trialscond, settings.num_setsizes, settings.num_days))

    # loop through days
    for day_count, day_val in enumerate(settings.daysuse):
        # get file names
        bids = helper.BIDS_FileNaming(sub_val, settings, day_val)

        # decide which file to use
        possiblefiles = []
        filesizes = []
        for filesfound in bids.direct_data_behave.glob(bids.filename_vissearch + "*.mat"):
            filesizes.append(filesfound.stat().st_size)
            possiblefiles.append(filesfound)

        file2useIDX = np.argmax(
            filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
        file2use = possiblefiles[file2useIDX]

        # load data
        F = h5py.File(file2use, 'r')
        print(list(F.keys()))

        for setcount, set_val in enumerate(settings.string_setsize):
            # reaction time data - exclude outliers
            tmp = np.array(F['RT'][set_val])

            if np.any(np.logical_not(np.isnan(tmp))):
                outliers = tmp > (np.nanmean(tmp) + np.nanstd(tmp) * 3)  # exclude data points 3 times greater than the mean
                tmp[outliers] = np.nan

                rt_vissearch[:, setcount, day_count] = tmp

            # Accuracy data - exclude outliers
            tmp = np.array(F['ACC'][set_val])
            tmp[outliers] = np.nan
            acc_vissearch[:, setcount, day_count] = tmp

        if np.nanmean(acc_vissearch[:, :, day_count] ) < 0.4:
            acc_vissearch[:, :, day_count] = np.abs(acc_vissearch[:, :, day_count] - 1)
            rt_vissearch[:, :, day_count] = np.nan

    # plot accuracy results
    meanacc = np.nanmean(acc_vissearch, axis=0) * 100

    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_prepost
    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, meanacc[0, :], width, label=settings.string_setsize[0], facecolor=settings.lightteal)
    plt.bar(x, meanacc[1, :], width, label=settings.string_setsize[1], facecolor=settings.medteal)
    plt.bar(x + width, meanacc[2, :], width, label=settings.string_setsize[2], facecolor=settings.darkteal)

    plt.ylim([70, 110])
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = bids.substring + ' Visual Search Accuracy train ' + settings.string_attntrained[settings.attntrained]
    plt.title(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # plot reaction time results
    meanrt = np.nanmean(rt_vissearch, axis=0)
    if ~np.any(np.isnan(meanrt)):
        fig, ax = plt.subplots(figsize=(12, 5))

        x = np.arange(len(labels))
        colors = [settings.lightteal, settings.medteal, settings.darkteal]

        for testday in np.arange(2):
            for setcount, set_val in enumerate(settings.string_setsize):
                data = rt_vissearch[:, setcount, testday]
                data = data[~np.isnan(data)]
                violin = ax.violinplot(dataset=data, positions=[setcount + 4 * testday], showmeans=False, showmedians=False,
                                       showextrema=False)

                for pc in violin['bodies']:
                    pc.set_facecolor(colors[setcount])
                    pc.set_edgecolor('black')
                    pc.set_alpha(1)

                quartile1, medians, quartile3 = np.percentile(data, [25, 50, 75])
                whiskersMin, whiskersMax = np.min(data), np.max(data)

                ax.scatter([setcount + 4 * testday], medians, marker='o', color='white', s=30, zorder=3)
                ax.vlines([setcount + 4 * testday], quartile1, quartile3, color='k', linestyle='-', lw=5)
                ax.vlines([setcount + 4 * testday], whiskersMin, whiskersMax, color='k', linestyle='-', lw=1)

        labels = ["", settings.string_prepost[0] + ' ' + settings.string_setsize[0],
                  settings.string_prepost[0] + ' ' + settings.string_setsize[1],
                  settings.string_prepost[0] + ' ' + settings.string_setsize[2], "",
                  settings.string_prepost[1] + ' ' + settings.string_setsize[0],
                  settings.string_prepost[1] + ' ' + settings.string_setsize[1],
                  settings.string_prepost[1] + ' ' + settings.string_setsize[2]
                  ]
        ax.set_xticklabels(labels)

        plt.ylabel('Reaction Time (s)')
        # plt.legend()
        ax.set_frame_on(False)

        titlestring = bids.substring + ' Visual Search Reaction Time train ' + settings.string_attntrained[
            settings.attntrained]
        plt.title(titlestring)
        plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # save results
    np.savez(bids.direct_results / Path(bids.substring + "visual_search_results"),
             meanacc=meanacc, meanrt=meanrt, acc_vissearch=acc_vissearch, rt_vissearch=rt_vissearch)


def collate_visualsearchtask(settings):
    print('Collating Visual Search Task')

    # Collate across group
    import seaborn as sns
    import pandas as pd

    # preallocate
    num_subs = np.zeros((settings.num_attnstates + 1))
    daystrings = []
    attnstrings = []
    setsizestrings = []
    substring = []
    accuracy_compare = []
    rt_compare = []

    meansubstring = []
    meandaystrings = []
    meanattnstrings = []
    meanaccuracy_compare = []
    meanrt_compare = []

    print('Collating Visual Search Task for space and feature train')
    for attntrained in np.arange(settings.num_attnstates + 1):  # cycle trough space and feature train groups

        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_visualsearchtask()

        # correct for lost data
        if attntrained == 2:  # correct for lost data for sub 90 (space train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([27, 14])))
            settings.num_subs = settings.num_subs - 2

        if attntrained == 1:  # correct for lost data for sub 89 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([89, 80, 110, 116, 23, 73])))
            settings.num_subs = settings.num_subs - 6

        if attntrained == 0:  # correct for lost data for sub 90 (space train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([90, 84, 107, 104, 118])))
            settings.num_subs = settings.num_subs - 5


        num_subs[attntrained] = settings.num_subs
        mean_acc_all = np.empty((settings.num_setsizes, settings.num_days, settings.num_subs))
        mean_rt_all = np.empty((settings.num_setsizes, settings.num_days, settings.num_subs))
        substring_short = []
        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "visual_search_results.npz"),
                              allow_pickle=True)  # saved vars: meanacc=meanacc, meanrt=meanrt, acc_vissearch=acc_nback, rt_vissearch=rt_nback

            # store results temporarily
            mean_acc_all[:, :, sub_count] = results['meanacc']
            mean_rt_all[:, :, sub_count] = results['meanrt']

            substring_short = np.concatenate((substring_short, ['sub' + str(sub_count + attntrained * 37)]))

        # store results for attention condition
        tmp = np.concatenate((substring_short, substring_short))
        tmp2 = np.concatenate((tmp, tmp))
        tmp2 = np.concatenate((tmp2, tmp))
        substring = np.concatenate((substring, tmp2))

        tmp = [settings.string_prepost[0]] * (settings.num_subs * settings.num_setsizes) + [
            settings.string_prepost[1]] * (settings.num_subs * settings.num_setsizes)
        daystrings = np.concatenate((daystrings, tmp))  # pretrain then postrain

        tmp = [settings.string_setsize[0]] * (settings.num_subs) + [settings.string_setsize[1]] * (
            settings.num_subs) + [settings.string_setsize[2]] * (settings.num_subs)
        setsizestrings = np.concatenate(
            (setsizestrings, tmp, tmp))  # Each setsize for each subject, repeated for the two testdays

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days * settings.num_setsizes
        attnstrings = np.concatenate(
            (attnstrings, tmp))  # All setsizes and subjects and days are the same attention trained

        tmp = np.concatenate((mean_acc_all[0, 0, :], mean_acc_all[1, 0, :], mean_acc_all[2, 0, :],
                              mean_acc_all[0, 1, :], mean_acc_all[1, 1, :], mean_acc_all[2, 1, :]))
        accuracy_compare = np.concatenate((accuracy_compare, tmp))

        tmp = np.concatenate((mean_rt_all[0, 0, :], mean_rt_all[1, 0, :], mean_rt_all[2, 0, :],
                              mean_rt_all[0, 1, :], mean_rt_all[1, 1, :], mean_rt_all[2, 1, :]))
        rt_compare = np.concatenate((rt_compare, tmp))

        # store results for attention condition - mean across set size conditions
        tmp = np.concatenate((substring_short, substring_short))
        meansubstring = np.concatenate((meansubstring, tmp))

        tmp = [settings.string_prepost[0]] * settings.num_subs + [settings.string_prepost[1]] * settings.num_subs
        meandaystrings = np.concatenate((meandaystrings, tmp))

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days
        meanattnstrings = np.concatenate((meanattnstrings, tmp))

        mean_acc_all2 = np.mean(mean_acc_all, axis=0)
        tmp = np.concatenate((mean_acc_all2[0, :], mean_acc_all2[1, :]))
        meanaccuracy_compare = np.concatenate((meanaccuracy_compare, tmp))

        # correct for overlarge or oversmall RTs
        mean_rt_all[mean_rt_all > 3] = np.nan
        mean_rt_all[mean_rt_all < 0.01] = np.nan
        mean_rt_all2 = np.nanmean(mean_rt_all, axis=0)
        tmp = np.concatenate((mean_rt_all2[0, :], mean_rt_all2[1, :]))
        meanrt_compare = np.concatenate((meanrt_compare, tmp))

    # create the data frames for accuracy and reaction time data
    data = {'SubID': substring, 'Testday': daystrings, 'Attention Trained': attnstrings, 'Set Size': setsizestrings,
            'Accuracy (%)': accuracy_compare, 'Reaction Time (s)': rt_compare}
    df_acc = pd.DataFrame(data)
    df_acc.to_csv(bids.direct_results_group_compare / Path("VISSEARCH_behaveresults_ALL.csv"), index=False)

    data = {'SubID': substring, 'Testday': daystrings, 'Attention Trained': attnstrings, 'Set Size': setsizestrings,
            'Reaction Time (s)': rt_compare}
    df_rt = pd.DataFrame(data)

    # correct for missing data
    df_rt = df_rt[df_rt["Reaction Time (s)"] < 3]

    # correct for tiny data
    df_rt = df_rt[df_rt["Reaction Time (s)"] > 0.01]

    # plot results
    fig, (ax1) = plt.subplots(1, 3, figsize=(12, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = [settings.lightteal, settings.medteal]

    for size, setsize in enumerate(settings.string_setsize):
        sns.swarmplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", dodge=True, data=df_rt[df_rt["Set Size"].isin([setsize])], color="0", alpha=0.3, ax=ax1[size])
        sns.violinplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", data=df_rt[df_rt["Set Size"].isin([setsize])], palette=sns.color_palette(colors), style="ticks", ax=ax1[size], split=False)

        handles, labels = ax1[size].get_legend_handles_labels()
        ax1[size].legend(handles[:2], labels[:2])

        ax1[size].spines['top'].set_visible(False)
        ax1[size].spines['right'].set_visible(False)

        ax1[size].set_title(setsize)
        ax1[size].set_ylim([0.4, 2.0])
    titlestring = 'Visual Search Results Compare Training'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')





    # Plot average over set sizes
    # create the data frames for accuracy and reaction time data
    data = {'SubID': meansubstring, 'Testday': meandaystrings, 'Attention Trained': meanattnstrings,
            'Accuracy (%)': meanaccuracy_compare, 'Reaction Time (s)': meanrt_compare}
    df_acc_SSmean = pd.DataFrame(data)

    # df_acc_SSmean.loc[df_acc_SSmean['Accuracy (%)']<60,:]

    ## Plot Accuracy to check that everythings ok
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.lightteal, settings.medteal]
    sns.swarmplot(x="Attention Trained", y="Accuracy (%)", hue="Testday", dodge=True, data=df_acc_SSmean, color="0", alpha=0.3, ax=ax1)
    sns.violinplot(x="Attention Trained", y="Accuracy (%)", hue="Testday", data=df_acc_SSmean,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=False)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], labels[:2])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Visual Search Accuracy by group'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    ## Plot Reaction time to check that everythings ok
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.lightteal, settings.medteal]
    sns.swarmplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", dodge=True, data=df_acc_SSmean, color="0", alpha=0.3, ax=ax1)
    sns.violinplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", data=df_acc_SSmean,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=False)
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[:2], labels[:2])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Visual Search Reaction Time by group'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    # Calculate training effects
    idx_d1 = df_acc_SSmean["Testday"] == "pre-training"
    idx_d4 = df_acc_SSmean["Testday"] == "post-training"

    tmpd4 = df_acc_SSmean[idx_d4].reset_index()
    tmpd1 = df_acc_SSmean[idx_d1].reset_index()

    df_behtraineffects = tmpd4[["SubID", "Attention Trained"]].copy()

    df_behtraineffects["∆Accuracy (%)"] = tmpd4['Accuracy (%)'] - tmpd1['Accuracy (%)']
    df_behtraineffects["∆Reaction Time (s)"] = tmpd4['Reaction Time (s)'] - tmpd1['Reaction Time (s)']

    # Plot accuracy and reaction time training effects
    colors = [settings.yellow, settings.orange, settings.red]
    fig, ax = plt.subplots(2, 1, figsize=(6, 15))

    measurestrings = ["∆Accuracy (%)", "∆Reaction Time (s)"]
    for i in np.arange(2):
        sns.swarmplot(x="Attention Trained", y=measurestrings[i], data=df_behtraineffects, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Trained", y=measurestrings[i], data=df_behtraineffects, palette=sns.color_palette(colors), style="ticks",
                       ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(measurestrings[i])

    titlestring = "Visual Search task training effects"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

# save out
    # df_rt_SSmean.to_pickle(bids.direct_results_group / Path("group_visualsearch.pkl"))

