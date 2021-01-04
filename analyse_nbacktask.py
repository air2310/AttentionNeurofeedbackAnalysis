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

    # Plot N-back pre vs. post Results

    import seaborn as sns
    import pandas as pd

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    sns.set(style="ticks")
    colors = ["#112F41", "#4CB99F"]
    # Accuracy

    data = {'Testday': settings.string_prepost,
            'Data': [np.mean(np.logical_or(acc_nback[:, 0] == 1, acc_nback[:, 0] == 3)),
                     np.mean(np.logical_or(acc_nback[:, 1] == 1, acc_nback[:, 1] == 3))]
            }
    meanacc = [np.mean(np.logical_or(acc_nback[:, 0] == 1, acc_nback[:, 0] == 3)),
               np.mean(np.logical_or(acc_nback[:, 1] == 1, acc_nback[:, 1] == 3))]
    df = pd.DataFrame(data)

    # Grouped violinplot
    sns.barplot(x="Testday", y="Data", data=df, palette=sns.color_palette(colors), ax=ax1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_ylabel('Accuracy (%)')

    ax1.set_title("Accuracy")

    # Reaction time
    data = {'Testday': [settings.string_prepost[0]] * settings.num_trials + [
        settings.string_prepost[1]] * settings.num_trials,
            'Data': np.concatenate((rt_nback[:, 0], rt_nback[:, 1]))
            }
    meanrt = np.nanmean(rt_nback, axis=0)
    df = pd.DataFrame(data)

    # Grouped violinplot

    sns.violinplot(x="Testday", y="Data", data=df, palette=sns.color_palette(colors), style="ticks", ax=ax2)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_ylabel('Reaction Time (s)')
    ax2.set_title("Reaction time")

    titlestring = bids.substring + ' Nback Results train ' + settings.string_attntrained[
        settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # save results
    np.savez(bids.direct_results / Path(bids.substring + "Nback_results"),
             meanacc=meanacc, meanrt=meanrt, acc_nback=acc_nback, rt_nback=rt_nback)


# old collation script - doesn't group space and feature data

    # print('Collating Visual Search Task')
    #
    # # get task specific settings
    # settings = settings.get_settings_nbacktask()
    #
    # # pre-allocate
    # num_subs = settings.num_subs
    # acc_nback_all = np.empty((settings.num_trials, settings.num_days, num_subs))
    # rt_nback_all = np.empty((settings.num_trials, settings.num_days, num_subs))
    # mean_acc_all = np.empty((settings.num_days, num_subs))
    # mean_rt_all = np.empty((settings.num_days, num_subs))
    #
    # # iterate through subjects for individual subject analyses
    # for sub_count, sub_val in enumerate(settings.subsIDXcollate):
    #     # get directories and file names
    #     bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
    #     print(bids.substring)
    #
    #     # load results
    #     results = np.load(bids.direct_results / Path(bids.substring + "Nback_results.npz"),
    #                       allow_pickle=True)  #
    #     # saved vars: meanacc=meanacc, meanrt=meanrt, acc_vissearch=acc_nback, rt_vissearch=rt_nback
    #
    #     # store results
    #     acc_nback_all[ :, :, sub_count] = results['acc_vissearch']
    #     mean_acc_all[ :, sub_count] = results['meanacc']
    #
    #     rt_nback_all[ :, :, sub_count] = results['rt_vissearch']
    #     mean_rt_all[ :, sub_count] = results['meanrt']
    #
    #
    # import seaborn as sns
    # import pandas as pd
    #
    # fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    # sns.set(style="ticks")
    # colors = ["#112F41", "#4CB99F"]
    # # Accuracy
    #
    # data = {'Testday': [settings.string_prepost[0]] * num_subs + [
    #     settings.string_prepost[1]] * num_subs,
    #         'Accuracy (%)': np.concatenate((mean_acc_all[0, :], mean_acc_all[1, :]))
    #         }
    # df = pd.DataFrame(data)
    #
    # # Grouped violinplot
    # sns.violinplot(x="Testday", y="Accuracy (%)", data=df, palette=sns.color_palette(colors), style="ticks", ax=ax1)
    # ax1.spines['top'].set_visible(False)
    # ax1.spines['right'].set_visible(False)
    #
    # ax1.set_title("Accuracy")
    #
    # # Reaction time
    # data = {'Testday': [settings.string_prepost[0]] * num_subs + [
    #     settings.string_prepost[1]] * num_subs,
    #         'Reaction Time (s)': np.concatenate((mean_rt_all[0 ,: ], mean_rt_all[1 ,: ]))
    #         }
    # df = pd.DataFrame(data)
    #
    # # Grouped violinplot
    #
    # sns.violinplot(x="Testday", y="Reaction Time (s)", data=df, palette=sns.color_palette(colors), style="ticks", ax=ax2)
    # ax2.spines['top'].set_visible(False)
    # ax2.spines['right'].set_visible(False)
    #
    # ax2.set_title("Reaction time")
    #
    # titlestring = 'Nback Results train ' + settings.string_attntrained[
    #     settings.attntrained]
    # plt.suptitle(titlestring)

    # plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')