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

