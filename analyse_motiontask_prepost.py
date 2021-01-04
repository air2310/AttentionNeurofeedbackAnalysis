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


def run_old(settings, sub_val, test_train):
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
            tmp = moveonsets[correctmoveorder, TT].T
            moveframes = np.array([np.nan, np.nan])
            for ii in np.arange(len(tmp)):
                tmp2 = np.array( [ tmp[ii] +settings.responseperiod[0], tmp[ii] +settings.responseperiod[1] ] )
                moveframes = np.row_stack((moveframes, tmp2.T))
            moveframes = np.delete(moveframes, 0, axis=0) # delete the row we used to initialise

            # Gather responses from this trial
            trialresponses = np.reshape(np.where(responses[:,1]==TT), -1)
            trialresponses_accounted = np.zeros(len(trialresponses))

            # Allocate response accuracy and reaction time for this trial
            for ii in np.arange(len(correct)): # cycle through all potential targets
                # get eligible responses within trial response period
                idx_response = np.array(np.where(np.logical_and(responses[:,1]==TT, np.logical_and(responses[:,2] > moveframes[ii,0],  responses[:,2] < moveframes[ii,1]))))
                idx_response = np.reshape(idx_response, -1)

                # if any of the responses during this trial happened during the necessary time period after a target, mark it as accounted for
                trialresponses_accounted[np.squeeze(np.isin(trialresponses, idx_response))] = 1

                # Fill out accuracy data - if no response to this target during the response period.
                if len(idx_response)==0: #if no response to this target during the response period.
                    resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_miss))
                    resp_reactiontime = np.row_stack(( resp_reactiontime, np.nan))
                    resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                elif len(idx_response) == 1:  # Correct or incorrect response?
                    if responses[idx_response, 0] == correct[ii]+1: # Correct Response!
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_correct))
                    else:
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_incorrect))

                    # tmp = responsetime[TT, responses[idx_response,2]] - moveonsets[np.reshape(correctmoveorder, -1)[ii],TT]/settings.mon_ref
                    tmp = responses[idx_response,2]/settings.mon_ref - moveonsets[np.reshape(correctmoveorder, -1)[ii], TT] / settings.mon_ref

                    # print(responses[idx_response,2]/settings.mon_ref-responsetime[TT, responses[idx_response,2]] )
                    # if (tmp > 1.75):
                    #     print("oh shit 1")


                    resp_reactiontime = np.row_stack((resp_reactiontime, tmp))
                    resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                elif len(idx_response) > 1:  # False alarm somewhere
                    idx_correct = np.where(np.squeeze(responses[idx_response, 0] == correct[ii]))
                    idx_falsealarm = np.where(np.squeeze(responses[idx_response, 0] != correct[ii]))

                    # Accuracy
                    tmp = np.ones(len(idx_response))*np.nan # create array of NaNs of length of number of responses
                    tmp[ idx_correct ] = settings.responseopts_correct # mark correct answers as correct
                    tmp[ idx_falsealarm ] = settings.responseopts_falsealarm  # mark incorrect answers as false alarms

                    resp_accuracy = np.row_stack((resp_accuracy, np.expand_dims(tmp, axis = 1)) )# stack to accuracy results

                    # Response Time and Trial Attention type
                    tmp = np.ones(len(idx_response)) * np.nan  # create array of NaNs of length of number of responses

                    # tmp[idx_correct] = responsetime[TT, responses[idx_response[idx_correct],2]] - moveonsets[np.reshape(correctmoveorder, -1)[ii],TT]/settings.mon_ref
                    tmp[idx_correct] =  responses[idx_response[idx_correct], 2] / settings.mon_ref - moveonsets[np.reshape(correctmoveorder, -1)[ii], TT] / settings.mon_ref

                    # responses[idx_response, 2] / settings.mon_ref # looks like the timer sometimes had a weird error where it is seconds of from the frames - was watching the frame rates for every subject and have never seen any seconds long frames like this, so I'm taking the frame response as the ground truth
                    # if (any(tmp > 1.75)):
                    #     print("oh shit 2")

                    resp_reactiontime = np.row_stack((resp_reactiontime,  np.expand_dims(tmp, axis = 1)))

                    # trial attention type
                    tmp = np.ones(len(idx_response)) * trialattntype[TT]
                    resp_trialattntype = np.row_stack((resp_trialattntype,  np.expand_dims(tmp, axis = 1)))

            if any(trialresponses_accounted==0): # if there are responses that hapended outside of the response periods for the targets
                num_falsealarms = sum(trialresponses_accounted==0)

                tmp = np.ones(num_falsealarms) * settings.responseopts_falsealarm
                resp_accuracy = np.row_stack((resp_accuracy, np.expand_dims(tmp, axis=1)))  # stack false alarms to accuracy results

                tmp = np.ones(num_falsealarms) * np.nan
                resp_reactiontime = np.row_stack((resp_reactiontime, np.expand_dims(tmp, axis=1)))# stack NaNs to RT results

                tmp = np.ones(num_falsealarms) * trialattntype[TT] # stack trial attention types
                resp_trialattntype = np.row_stack((resp_trialattntype, np.expand_dims(tmp, axis=1)))

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
        sns.violinplot(x="Attention Type", y="Reaction Time", hue="Testday",data=df_behavedata, palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="quartile")
    else:
        colors = ["#F2B035", "#EC553A", "#C11F3A"] # yellow, orange, red
        sns.violinplot(x="Testday", y="Reaction Time",data=df_behavedata,
                       palette=sns.color_palette(colors), style="ticks", ax=ax1)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring =  bids.substring + ' Motion Task RT by Day ' +  settings.string_testtrain[test_train]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # next up, plot accuracy results and then save data. Functionalise then summarise!

    # copy data to new accuracy data frame
    df_accuracy = df_behavedata[['Testday', 'Attention Type']].copy()


    # fill in the columns
    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_correct])] = 1
    df_accuracy.loc[:,'correct'] = pd.DataFrame({'correct': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_incorrect])] = 1
    df_accuracy.loc[:,'incorrect'] = pd.DataFrame({'incorrect': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_falsealarm])] = 1
    df_accuracy.loc[:,'falsealarm'] = pd.DataFrame({'falsealarm': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_miss])] = 1
    df_accuracy.loc[:,'miss'] = pd.DataFrame({'miss': tmp})

    # count values for each accuracy type
    # df_accuracy.set_index(['Testday', 'Attention Type'], inplace=True)
    acc_count = df_accuracy.groupby(['Testday', 'Attention Type']).sum()


    # Normalise
    totals = acc_count["miss"]+ acc_count['incorrect'] + acc_count["falsealarm"]+ acc_count['correct']
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
    colors = ["#4EB99F", "#C11F3A", "#F2B035", "#EC553A"] # light teal,red, yellow, orrange

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


    titlestring = bids.substring + ' Motion Task Accuracy by Day ' +  settings.string_testtrain[test_train]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

    # save results.
    df_accuracy.to_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_acc_" +  settings.string_testtrain[test_train] + ".pkl"))
    df_behavedata.to_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_allbehave_" +  settings.string_testtrain[test_train] + ".pkl"))
