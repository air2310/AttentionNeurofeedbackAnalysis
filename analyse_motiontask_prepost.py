# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

import helperfunctions_ATTNNF as helper


def run(settings, sub_val, test_train):
    #get task specific settings
    if (test_train == 0):
        settings = settings.get_settings_behave_prepost()
    else:
        settings = settings.get_settings_behave_duringNF()

    # pre-allocate
    # acc_nback = np.empty((settings.num_trials, settings.num_days))
    # rt_nback = np.empty((settings.num_trials, settings.num_days))

    # loop through days
    for day_count, day_val in enumerate(settings.daysuse):
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
        else:
            if (day_count==0):
                 df_behavedata=[] # assign df_behave data as empty if this happens on the first day
            continue

        # load data
        F = h5py.File(file2use, 'r')
        # print(list(F.keys())) #'DATA', 'RESPONSE', 'RESPONSETIME', 'trialtype'

        ### get variables of interest
        # Response Data
        response_raw = np.array(F['RESPONSE'])
        responsetime = np.array(F['RESPONSETIME'])

        # Experiment Settings
        trialattntype = np.squeeze(np.array(F['DATA']['trialattntype']))
        moveonsets = np.array(F['DATA']['MOVEONSETS'])
        directionchanges = np.array(F['DATA']['DirChanges__Moveorder'])
        moveorder = np.squeeze(np.array(F['DATA']['Moveorder']))

        num_motionepochspertrial = 5
        # process responses
        response_diff_idx = np.column_stack((np.zeros((settings.num_trials,)).T, np.diff(response_raw, axis=1))) # find changes in trials x frames response variable
        responses = np.array([np.nan, np.nan, np.nan]) # start of responses array

        # Find responses on each trial
        for TT in np.arange(settings.num_trials):
            idx_trialresponses = np.asarray(np.where(response_diff_idx[TT,:]>0)) # Where were the responses this trial?

            if (idx_trialresponses.size>0): # if thre were any responses this trial
                for ii in np.arange(idx_trialresponses.shape[1]): # loop through those responses
                    # stack response, trial, and index of all responses together
                    idx = idx_trialresponses.item(ii)
                    tmp = np.array([response_raw.item((TT,idx)), TT, idx])
                    responses = np.row_stack((responses, tmp))

        tmp = np.delete(responses, 0,0) # delete Nans at the begining
        responses = tmp.astype(int) # convert back to integers

        # Score behaviour ----------------------------------------------------------------------------------------

        resp_accuracy = np.array(np.nan)
        resp_reactiontime = np.array(np.nan)
        resp_trialattntype = np.array(np.nan)
        # resp_daystring = np.array([np.nan])

        for TT in np.arange(settings.num_trials):
            # Define when the targets happened on this trial

            # for moveorder, coding is in terms of the cue so: 1 = Cued Feature, Cued Space, 2 = Uncued Feature, Cued Space, 3 = Cued Feature, Uncued Space, 4 = Uncued Feature, Uncued Space.
            # For the pre and post training task, this coding is a little irrelevant, as we get either a space or a feature cue, not both (like we do during neurofeedback). However, we just used the same coding to make the
            # task easier to write up. Therefore, either 1 or 3 could mean the cued feature for feature cues, and either 1 or 2 could mean the cued space for space cues.
            # if trial attntype == 3 - during NF, only want moveorder = 1.
            if (trialattntype[TT] == 1): # Feature
                correctmoveorder = np.where(np.logical_or(moveorder[:,TT] == 1, moveorder[:, TT] == 3))

            if (trialattntype[TT] == 2): # Space
                correctmoveorder = np.where(np.logical_or(moveorder[:,TT] == 1, moveorder[:, TT] ==2))

            if (trialattntype[TT] == 3):  # Space
                correctmoveorder = np.where(moveorder[:, TT] == 1)

            # Define correct answers for this trial
            correct = directionchanges[correctmoveorder, TT].T
            for ii in np.arange(len(correct)):
                correct[ii] = np.where(settings.directions==correct[ii])



            # define period in which correct answers happen
            # tmp = moveonsets[correctmoveorder, TT].T
            tmp = moveonsets[:, TT].T
            moveframes = np.array([np.nan, np.nan])
            for ii in np.arange(len(tmp)):
                tmp2 = np.array( [ tmp[ii] +settings.responseperiod[0], tmp[ii] +settings.responseperiod[1] ] )
                moveframes = np.row_stack((moveframes, tmp2.T))
            moveframes = np.delete(moveframes, 0, axis=0) # delete the row we used to initialise

            # # Gather responses from this trial
            trialresponses = np.reshape(np.where(responses[:,1]==TT), -1)
            # trialresponses_accounted = np.zeros(len(trialresponses))

            # Allocate response accuracy and reaction time for this trial
            for ii in np.arange(num_motionepochspertrial): # cycle through all potential targets
                # get eligible responses within trial response period
                idx_response = np.array(np.where(np.logical_and(responses[:,1]==TT, np.logical_and(responses[:,2] > moveframes[ii,0],  responses[:,2] < moveframes[ii,1]))))
                idx_response = np.reshape(idx_response, -1)

                # define what the correct response would have been
                correct = np.squeeze(np.where(settings.directions == directionchanges[ii, TT]))

                # Is this a target?
                if np.any(ii == np.squeeze(correctmoveorder)): # is this a target event? if so:

                    # Miss! - if no response to this target during the response period, allocate it as a miss.
                    if len(idx_response) == 0:  # if no response to this target during the response period.
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_miss))  # this target was missed
                        resp_reactiontime = np.row_stack((resp_reactiontime, np.nan))
                        resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                    # Correct or incorrect response?
                    elif len(idx_response) == 1:
                        # accuracy
                        if responses[idx_response, 0] == correct + 1:  # Correct Response!
                            # Accuracy
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_correct))

                            # reaction time
                            tmp = responses[idx_response, 2] / settings.mon_ref - moveonsets[ii, TT] / settings.mon_ref
                            resp_reactiontime = np.row_stack((resp_reactiontime, tmp))

                        else: # incorrect response
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_incorrect))
                            resp_reactiontime = np.row_stack((resp_reactiontime, np.nan)) # don't keep track of these NAN RTs


                        # trial attention type (to store)
                        resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                    # Multiple responses to target!
                    elif len(idx_response) > 1:
                        print("double response!")
                        # find out if any of them are correct, if so, take the first correct one, if not, mark as incorrect.
                        idx_correct = np.where(np.squeeze(responses[idx_response, 0] == correct))

                        if np.shape(idx_correct)[1] > 0: # were any correct
                            # Accuracy
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_correct))

                            # reaction time
                            tmp = responses[idx_response[idx_correct[0][0]], 2] / settings.mon_ref - moveonsets[ii, TT] / settings.mon_ref # use the reaction time for the first correct response
                            resp_reactiontime = np.row_stack((resp_reactiontime, tmp))

                        else: # mark as incorrect
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_incorrect))
                            resp_reactiontime = np.row_stack((resp_reactiontime, np.nan))  # don't keep track of these NAN RTs

                        resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                else: # non target
                    # If no response to this target during the response period, allocate it as a correct reject.
                    if len(idx_response) == 0:  # if no response to this target during the response period.
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_correctreject))  # Correct rejection!
                        resp_reactiontime = np.row_stack((resp_reactiontime, np.nan))
                        resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                    # Multiple responses to target!
                    elif len(idx_response) > 1:
                        print("double false alarm response!")
                        idx_correct = np.where(np.squeeze(responses[idx_response, 0] == correct))

                        if np.shape(idx_correct)[1] > 0:  # were any correct
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm))
                        else:  # mark as incorrect
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm_incorrect))

                        resp_reactiontime = np.row_stack((resp_reactiontime, np.nan))  # add NAN to RT vector
                        resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                    else: # any responses in this period are false alarms.
                        # (TODO) if neurofeedback - store as false alarm, with info about what sort of false alarm for future breakdown.

                        if responses[idx_response, 0] == correct + 1:  # Responding correctly to Distractor's motion direction!
                            # Accuracy
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm))

                        else:  # Responding incorrectly to Distractor's motion direction!
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm_incorrect))

                        # resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm))  # stack FA to accuracy results
                        resp_reactiontime = np.row_stack((resp_reactiontime, np.nan))  # add NAN to RT vector
                        resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))  # trial attention type



        # next up! Stack across days, plot with seaborn :)

        # delete the nans
        resp_trialattntype = np.delete(resp_trialattntype, 0, axis=0)  # delete the row we used to initialise
        resp_reactiontime = np.delete(resp_reactiontime, 0, axis=0)  # delete the row we used to initialise
        resp_accuracy = np.delete(resp_accuracy, 0, axis=0)  # delete the row we used to initialise

        # gather number of responses for this day and generate day strings variable
        num_responses = len(resp_accuracy)
        daystrings = [settings.string_testday[day_val-1]] * num_responses

        # change trial attention type to strings
        resp_trialattntype =resp_trialattntype.astype(str)
        resp_trialattntype[resp_trialattntype=='1.0'] = settings.trialattntype[0]
        resp_trialattntype[resp_trialattntype == '2.0'] = settings.trialattntype[1]
        resp_trialattntype[resp_trialattntype == '3.0'] = settings.trialattntype[2]

        # list behavioural data
        behavedata = {'Testday': daystrings, 'Accuracy': np.squeeze(resp_accuracy),
                      'Reaction Time': np.squeeze(resp_reactiontime),
                      'Attention Type': np.squeeze(resp_trialattntype)}
        if (day_count == 0 or not any(df_behavedata)) :
            df_behavedata = pd.DataFrame(behavedata)

        else: # apend second day data to first day data
            df_tmp = pd.DataFrame(behavedata)

            # add day two data to the official data frame
            df_behavedata = df_behavedata.append(df_tmp, ignore_index=True)

    ########## Time to plot! ##########
    # Step 1: exclude reaction time outliers

    means = df_behavedata.groupby(['Testday', 'Attention Type']).mean()
    SDs = df_behavedata.groupby(['Testday', 'Attention Type']).std()

    # plot reaction time results

    # plot results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot


    if (test_train==0):
        colors = ["#F2B035", "#EC553A"]
        sns.violinplot(x="Attention Type", y="Reaction Time", hue="Testday",data=df_behavedata, palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")
    else:
        colors = ["#F2B035", "#EC553A", "#C11F3A"] # yellow, orange, red
        sns.violinplot(x="Testday", y="Reaction Time",data=df_behavedata,
                       palette=sns.color_palette(colors), style="ticks", ax=ax1, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring =  bids.substring + ' Motion Task RT by Day ' +  settings.string_testtrain[test_train]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # next up, plot accuracy results and then save data. Functionalise then summarise!

    # copy data to new accuracy data frame
    df_accuracy_targets = df_behavedata[['Testday', 'Attention Type']].copy()

    # fill in the columns
    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_correct])] = 1
    df_accuracy_targets.loc[:,'correct'] = pd.DataFrame({'correct': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_incorrect])] = 1
    df_accuracy_targets.loc[:,'incorrect'] = pd.DataFrame({'incorrect': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_miss])] = 1
    df_accuracy_targets.loc[:,'miss'] = pd.DataFrame({'miss': tmp})

    # tmp = np.zeros(df_behavedata.__len__())
    # tmp[df_behavedata["Accuracy"].isin([settings.responseopts_falsealarm])] = 1
    # df_accuracy_targets.loc[:,'falsealarm'] = pd.DataFrame({'falsealarm': tmp})

    # count values for each accuracy type
    # df_accuracy.set_index(['Testday', 'Attention Type'], inplace=True)
    acc_count = df_accuracy_targets.groupby(['Testday', 'Attention Type']).sum()


    # Normalise
    totals = acc_count["miss"]+ acc_count['incorrect'] + acc_count['correct']
    acc_count = acc_count.div(totals, axis=0)*100

    # Organise for plotting
    acc_count2 = acc_count.swaplevel()
    acc_count2 = acc_count2.T

    # plot!
    if (test_train == 0):
        fig, (ax) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))

    # style
    colors = [settings.lightteal, settings.red, settings.yellow] # light teal,red, yellow, orrange

    sns.set(style="ticks", palette=sns.color_palette(colors))

    if (test_train == 0):
        # Reaction time Grouped violinplot
        for attn, attn_val in enumerate(settings.string_attntype):
            acc_count2[attn_val].T.plot(kind='bar', stacked=True,  ax=ax[attn])

            ax[attn].spines['top'].set_visible(False)
            ax[attn].spines['right'].set_visible(False)

            ax[attn].set_ylabel('Response %')
            ax[attn].set_title(attn_val)
    else:
        acc_count2['Both'].T.plot(kind='bar', stacked=True, ax=ax)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel('Response %')


    titlestring = bids.substring + ' Motion Task Target Response Accuracy by Day ' +  settings.string_testtrain[test_train]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # The same thing, but for distractors

    # copy data to new accuracy data frame
    df_accuracy_distract = df_behavedata[['Testday', 'Attention Type']].copy()

    # fill in the columns
    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_falsealarm])] = 1
    df_accuracy_distract.loc[:,'falsealarm'] = pd.DataFrame({'falsealarm': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_falsealarm_incorrect])] = 1
    df_accuracy_distract.loc[:, 'falsealarm_incorrect'] = pd.DataFrame({'falsealarm_incorrect': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_correctreject])] = 1
    df_accuracy_distract.loc[:,'correctreject'] = pd.DataFrame({'correctreject': tmp})

    # count values for each accuracy type
    acc_count = df_accuracy_distract.groupby(['Testday', 'Attention Type']).sum()

    # Normalise
    totals = acc_count["falsealarm"]+ acc_count['correctreject'] + acc_count['falsealarm_incorrect']
    acc_count = acc_count.div(totals, axis=0)*100

    # Organise for plotting
    acc_count2 = acc_count.swaplevel()
    acc_count2 = acc_count2.T

    # plot!
    if (test_train == 0):
        fig, (ax) = plt.subplots(1, 2, figsize=(12, 6))
    else:
        fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))

    # style
    colors = [settings.orange,  settings.darkteal,settings.medteal] # light teal,red, yellow, orrange

    sns.set(style="ticks", palette=sns.color_palette(colors))

    if (test_train == 0):
        # Reaction time Grouped violinplot
        for attn, attn_val in enumerate(settings.string_attntype):
            acc_count2[attn_val].T.plot(kind='bar', stacked=True,  ax=ax[attn])

            ax[attn].spines['top'].set_visible(False)
            ax[attn].spines['right'].set_visible(False)

            ax[attn].set_ylabel('Response %')
            ax[attn].set_title(attn_val)
    else:
        acc_count2['Both'].T.plot(kind='bar', stacked=True, ax=ax)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel('Response %')


    titlestring = bids.substring + ' Motion Task Distractor Response Accuracy by Day ' +  settings.string_testtrain[test_train]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')



    # save results.
    df_accuracy_targets.to_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_acctarget_" + settings.string_testtrain[test_train] + ".pkl"))
    df_accuracy_distract.to_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_accdistract_" +  settings.string_testtrain[test_train] + ".pkl"))
    df_behavedata.to_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_allbehave_" +  settings.string_testtrain[test_train] + ".pkl"))


def collate_behaviour_prepost(settings):
    print('Collating Motion Task Behaviour')

    # get task specific settings
    settings = settings.get_settings_behave_prepost()

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        accdat_targ_sub = pd.read_pickle(bids.direct_results / Path(
            bids.substring + "motiondiscrim_acctarget_" + settings.string_testtrain[0] + ".pkl"))
        accdat_dist_sub = pd.read_pickle(bids.direct_results / Path(
            bids.substring + "motiondiscrim_accdistract_" + settings.string_testtrain[0] + ".pkl"))
        behdat_sub = pd.read_pickle(bids.direct_results / Path(
            bids.substring + "motiondiscrim_allbehave_" + settings.string_testtrain[0] + ".pkl"))

        # get percentage of responses for each subject in each category - targets
        acc_targ_count = accdat_targ_sub.groupby(['Testday', 'Attention Type']).sum()
        totals = acc_targ_count["miss"] + acc_targ_count['incorrect'] + acc_targ_count['correct']
        acc_targ_count = acc_targ_count.div(totals, axis=0) * 100
        accdat_targ_sub = acc_targ_count.reset_index()

        # get percentage of responses for each subject in each category - distractors
        acc_dist_count = accdat_dist_sub.groupby(['Testday', 'Attention Type']).sum()
        totals = acc_dist_count["falsealarm"] + acc_dist_count['falsealarm_incorrect'] + acc_dist_count['correctreject']
        acc_dist_count = acc_dist_count.div(totals, axis=0) * 100
        accdat_dist_sub = acc_dist_count.reset_index()

        # Add Sub ID column
        accdat_targ_sub['subID'] = sub_count
        accdat_dist_sub['subID'] = sub_count
        behdat_sub['subID'] = sub_count

        accdat_targ_sub['subIDval'] = sub_val
        accdat_dist_sub['subIDval'] = sub_val
        behdat_sub['subIDval'] = sub_val

        # Stack across subjects
        if (sub_count == 0):  # First subject, no dataframe exists yet
            accdat_targ_all = accdat_targ_sub
            accdat_dist_all = accdat_dist_sub
            behdat_all = behdat_sub
        else:
            accdat_targ_all = accdat_targ_all.append(accdat_targ_sub,
                                                     ignore_index=True)  # ignore index just means the index will count all the way up
            accdat_dist_all = accdat_dist_all.append(accdat_dist_sub,
                                                     ignore_index=True)  # ignore index just means the index will count all the way up
            behdat_all = behdat_all.append(behdat_sub, ignore_index=True)

    # average and plot reaction time data
    behdat_all_avg = behdat_all.groupby(["subID", "Testday", "Attention Type"]).mean()
    behdat_all_avg = behdat_all_avg.reset_index()

    # get standard dev
    behdat_all_std = behdat_all.groupby(["subID", "Testday", "Attention Type"]).std()
    behdat_all_std = behdat_all_std.reset_index()

    # get inverse efficiency
    behdat_all_avg["InverseEfficiency"] = behdat_all_avg["Reaction Time"] / (accdat_targ_all["correct"] / 100)

    # plot results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="Reaction Time", hue="Testday", data=behdat_all_avg,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim([0.4, 1.2])

    titlestring = 'Motion Task RT by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot results - standard deviation
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="Reaction Time", hue="Testday", data=behdat_all_std,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([0.4, 1.2])

    titlestring = 'Motion Task RT Std Dev by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot results - inverse efficiency
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="InverseEfficiency", hue="Testday", data=behdat_all_avg,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([0.4, 1.2])

    titlestring = 'Motion Task RT Inverse Efficiency by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # Plot Accuracy Data

    ############## Get Sensitivity
    from scipy.stats import norm
    import math

    Z = norm.ppf  # percentile point function - normal distribution between 0 and 1.

    N_distractors = 336  # number of distractor events per day and condition.
    N_targets = 144  # number of target events per day and condition.

    dat = accdat_targ_all.loc[:, "correct"] / 100  # hitrate

    dat[dat == 0] = 1 / (2 * N_targets)  # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_targets)  # correct for zeros and ones (McMillan & Creelman, 2004)

    hitrate_zscore = Z(dat)

    dat = accdat_dist_all.loc[:, "falsealarm"] / 100

    dat[dat == 0] = 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)

    falsealarmrate_zscore = Z(dat)

    accdat_targ_all.loc[:, "Sensitivity"] = hitrate_zscore - falsealarmrate_zscore
    accdat_targ_all.loc[:, "Criterion"] = 0.5 * (hitrate_zscore + falsealarmrate_zscore)
    accdat_targ_all.loc[:, "LikelihoodRatio"] = accdat_targ_all.loc[:, "Sensitivity"] * accdat_targ_all.loc[:,
                                                                                        "Criterion"]

    # plot results - sensitivity
    plots = ["correct", "miss", "correctreject", "falsealarm"]

    fig, (ax1) = plt.subplots(2, 2, figsize=(10, 10))
    ax1 = ax1.flatten()
    sns.set(style="ticks")
    colors = ["#F2B035", "#EC553A"]
    # Reaction time Grouped violinplot
    for ii in np.arange(4):
        if ii < 2:
            sns.violinplot(x="Attention Type", y=plots[ii], hue="Testday", data=accdat_targ_all,
                           palette=sns.color_palette(colors), style="ticks", ax=ax1[ii], split=True, inner="stick")
        else:
            sns.violinplot(x="Attention Type", y=plots[ii], hue="Testday", data=accdat_dist_all,
                           palette=sns.color_palette(colors), style="ticks", ax=ax1[ii], split=True, inner="stick")

        ax1[ii].spines['top'].set_visible(False)
        ax1[ii].spines['right'].set_visible(False)

        ax1[ii].set_title(plots[ii])

        ax1[ii].set_ylim([0, 100])
    titlestring = 'Motion Task Accuracy by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot sensitivity results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="Sensitivity", hue="Testday", data=accdat_targ_all,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim([-1, 5])

    titlestring = 'Motion Task Sensitivity by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot Criterion results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="Criterion", hue="Testday", data=accdat_targ_all,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([-1,5])

    titlestring = 'Motion Task Criterion by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot Likelihood Ratio results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="LikelihoodRatio", hue="Testday", data=accdat_targ_all,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([-1,5])

    titlestring = 'Motion Task Likelihood Ratio by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # Save results out for collation later
    accdat_targ_all['RT'] = behdat_all_avg['Reaction Time']
    accdat_targ_all['RT_STD'] = behdat_all_std['Reaction Time']
    accdat_targ_all["InverseEfficiency"] = behdat_all_avg["InverseEfficiency"]
    accdat_targ_all.to_pickle(
        bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))
