# Import nescescary packages
from distutils.command.sdist import sdist
import numpy as np
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import helperfunctions_ATTNNF as helper
import CorrelationAnalyses as datacompare

def get_data(sub_val, settings, day_val, day_count):
    # get file names
    bids = helper.BIDS_FileNaming(sub_val, settings, day_val)

    # decide which file to use
    possiblefiles = []
    filesizes = []
    missingdat = 0
    for filesfound in bids.direct_data_behave.glob(bids.filename_behave + "*.mat"):
        filesizes.append(filesfound.stat().st_size)
        possiblefiles.append(filesfound)

    if any(filesizes):
        file2useIDX = np.argmax(filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
        file2use = possiblefiles[file2useIDX]
    else:
        if (day_count==0):
                df_behavedata=[] # assign df_behave data as empty if this happens on the first day
        missingdat=1

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

    return bids, responses, moveorder, moveonsets, responsetime, trialattntype, directionchanges, missingdat, num_motionepochspertrial

def score_behave(settings, responses, moveorder, moveonsets, trialattntype, directionchanges, num_motionepochspertrial, day_val):

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

                # Multiple responses to non-target!
                elif len(idx_response) > 1:
                    print("double false alarm response!")
                    idx_correct = np.where(np.squeeze(responses[idx_response, 0] == correct))

                    if np.shape(idx_correct)[1] > 0:  # were any correct
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm))
                    else:  # mark as incorrect
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm)) #_incorrect))

                    resp_reactiontime = np.row_stack((resp_reactiontime, np.nan))  # add NAN to RT vector
                    resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                else: # any responses in this period are false alarms.

                    if responses[idx_response, 0] == correct + 1:  # Responding correctly to Distractor's motion direction!
                        # Accuracy
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm))

                    else:  # Responding incorrectly to Distractor's motion direction!
                        resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm)) #_incorrect))

                    # resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_falsealarm))  # stack FA to accuracy results
                    resp_reactiontime = np.row_stack((resp_reactiontime, np.nan))  # add NAN to RT vector
                    resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))  # trial attention type

    # next up! Stack across days
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

    return behavedata


def plot_reactiontimes(df_behavedata, test_train, bids, settings):
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    if (test_train==0):
        colors = ["#F2B035", "#EC553A"]

        sns.swarmplot("Testday", y="Reaction Time", data=df_behavedata, ax=ax1, color=".2")

        sns.violinplot(x="Testday", y="Reaction Time", data=df_behavedata,
                           palette=sns.color_palette(colors), ax=ax1)

    else:
        colors = ["#F2B035", "#EC553A", "#C11F3A"] # yellow, orange, red
        sns.violinplot(x="Testday", y="Reaction Time",data=df_behavedata,
                       palette=sns.color_palette(colors), style="ticks", ax=ax1, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring =  bids.substring + ' Motion Task RT by Day cuemean' +  settings.string_testtrain[test_train]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')



def plot_stackedbarplot(acc_count, test_train, settings, bids):
        # Organise for plotting
    # plot!
    if (test_train == 0):
        fig, (ax) = plt.subplots(1, 1, figsize=(12, 6))
    else:
        fig, (ax) = plt.subplots(1, 1, figsize=(6, 6))

    # style
    colors = [settings.lightteal, settings.red, settings.yellow] # light teal,red, yellow, orrange

    sns.set(style="ticks", palette=sns.color_palette(colors))

    if (test_train == 0):
        # Reaction time Grouped violinplot
        acc_count.plot(kind='bar', stacked=True,  ax=ax)
    else:
        acc_count['Both'].plot(kind='bar', stacked=True, ax=ax)

    ax.set_ylabel('Response %')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    titlestring = bids.substring + ' Motion Task Target Response Accuracy by Day cuemean ' +  settings.string_testtrain[test_train]
    plt.suptitle(titlestring)
    # plt.savefig(bids.direct_results / Path(titlestring + '.png'), format='png')

def run(settings, sub_val, test_train):
    #get task specific settings
    if (test_train == 0):
        settings = settings.get_settings_behave_prepost()
    else:
        settings = settings.get_settings_behave_duringNF()

    # loop through days
    for day_count, day_val in enumerate(settings.daysuse):

        # Get data
        bids, responses, moveorder, moveonsets, responsetime, trialattntype, directionchanges, missingdat, num_motionepochspertrial = get_data(sub_val, settings, day_val, day_count)

        # If datafile is missing, don't try to analyse it. 
        if missingdat:
            continue
        
        # Score behaviour ----------------------------------------------------------------------------------------
        behavedata = score_behave(settings, responses, moveorder, moveonsets, trialattntype, directionchanges, num_motionepochspertrial, day_val)
        
        # Stack across days
        if (day_count == 0 or not any(df_behavedata)) :
            df_behavedata = pd.DataFrame(behavedata)

        else: # apend second day data to first day data
            df_behavedata = pd.concat((df_behavedata, pd.DataFrame(behavedata)), axis=0, ignore_index=True)
    
    ########## Time to plot! ##########
    # exclude reaction time outliers?
    means = df_behavedata.groupby(['Testday']).mean()
    SDs = df_behavedata.groupby(['Testday']).std()

    exclude = []
    for testday in means.index:
        M, SD = means["Reaction Time"][testday], SDs["Reaction Time"][testday]
        excluderange = [M - 2.5*SD, M + 2.5*SD]
        print(excluderange)
        dat = df_behavedata[df_behavedata['Testday'].isin([testday])]['Reaction Time']
        toexclude = (dat<excluderange[0] ) | (dat>excluderange[1] )
        exclude.extend(toexclude.index[toexclude].tolist())

    df_behavedata = df_behavedata.drop(exclude)

    # plot reaction times
    plot_reactiontimes(df_behavedata, test_train, bids, settings)

    # copy data to new accuracy data frame
    df_accuracy_targets = df_behavedata[['Testday']].copy()

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

    # count values for each accuracy type
    acc_count = df_accuracy_targets.groupby(['Testday']).sum()


    # Normalise
    targ_totals = acc_count["miss"]+ acc_count['incorrect'] + acc_count['correct']
    acc_count = acc_count.div(targ_totals, axis=0)*100

    # The same thing, but for distractors
    plot_stackedbarplot(acc_count, test_train, settings, bids)

    # copy data to new accuracy data frame
    df_accuracy_distract = df_behavedata[['Testday']].copy()

    # fill in the columns
    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_falsealarm])] = 1
    df_accuracy_distract.loc[:,'falsealarm'] = pd.DataFrame({'falsealarm': tmp})

    tmp = np.zeros(df_behavedata.__len__())
    tmp[df_behavedata["Accuracy"].isin([settings.responseopts_correctreject])] = 1
    df_accuracy_distract.loc[:,'correctreject'] = pd.DataFrame({'correctreject': tmp})

    # count values for each accuracy type
    inacc_count = df_accuracy_distract.groupby(['Testday']).sum()

    # Normalise
    dist_totals = inacc_count["falsealarm"]+ inacc_count['correctreject'] 
    inacc_count = inacc_count.div(dist_totals, axis=0)*100

    # The same thing, but for distractors
    plot_stackedbarplot(inacc_count, test_train, settings, bids)

    # Calculate sensitivity and shit. 

    ############## Get Sensitivity
    from scipy.stats import norm
    import math

    Z = norm.ppf  # percentile point function - normal distribution between 0 and 1.

    N_distractors = dist_totals  # number of distractor events per day and condition.
    N_targets =  targ_totals   # number of target events per day and condition.

    dat =acc_count["correct"] / 100  # hitrate

    dat[dat == 0] = 1 / (2 * N_targets)  # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_targets)  # correct for zeros and ones (McMillan & Creelman, 2004)

    hitrate_zscore = Z(dat)

    dat =inacc_count["falsealarm"] / 100

    dat[dat == 0] = 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)

    falsealarmrate_zscore = Z(dat)

    # results
    SubID = [settings.string_attntrained[settings.attntrained]  + str(sub_val), settings.string_attntrained[settings.attntrained]  + str(sub_val)]
    AttentionTrained = [settings.string_attntrained[settings.attntrained], settings.string_attntrained[settings.attntrained]]
    testdays = ["Day 1", "Day 4"]
    correct = [acc_count["correct"][0], acc_count["correct"][1]]
    incorrect = [acc_count["incorrect"][0], acc_count["incorrect"][1]]
    miss = [acc_count["miss"][0], acc_count["miss"][1]]
    falsealarm = [inacc_count["falsealarm"][0], inacc_count["falsealarm"][1]]
    sensitivity = hitrate_zscore - falsealarmrate_zscore
    criterion =  0.5 * (hitrate_zscore + falsealarmrate_zscore)
    LR = sensitivity * criterion
    RT = [df_behavedata[df_behavedata['Testday'].isin(["Day 1"])]['Reaction Time'].mean(), df_behavedata[df_behavedata['Testday'].isin(["Day 4"])]['Reaction Time'].mean()]
    RTSTD = [df_behavedata[df_behavedata['Testday'].isin(["Day 1"])]['Reaction Time'].std(), df_behavedata[df_behavedata['Testday'].isin(["Day 4"])]['Reaction Time'].std()]
    rejectsubject = [df_accuracy_targets.groupby(['Testday']).sum()['correct'][0]<30, df_accuracy_targets.groupby(['Testday']).sum()['correct'][1]<30]

    results = {'SubID':SubID, 
    'AttentionTrained': AttentionTrained,
    'Testday': testdays, 
    'correct': correct,
    'incorrect': incorrect,
    'miss':miss,
    'falsealarm':falsealarm,
    'sensitivity ': sensitivity ,
    'criterion':criterion,
    'LR' :LR,
    'RT':RT,
    'RTSTD':RTSTD, 
    'rejectsubject': rejectsubject}

    # save results.
    filename = bids.direct_results / Path(bids.substring + "motiondiscrim_cuemean_" + settings.string_testtrain[test_train] + ".npy")
    np.save(filename, results)










def collate_behaviour_prepost(settings):
    print('Collating Motion Task Behaviour')

    # get task specific settings
    settings = settings.get_settings_behave_prepost()
    data = {}

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        filename =  bids.direct_results / Path(bids.substring + "motiondiscrim_cuemean_" + settings.string_testtrain[0] + ".npy")
        perform_dict = np.load(filename,allow_pickle=True)[()]

        #   Append to list
        for key in perform_dict.keys():
            if not(isinstance(perform_dict[key], (list))):
                perform_dict[key] = perform_dict[key].tolist()
            try:                
                data[key].extend(perform_dict[key])
            except KeyError: # For the first subject, we don't have any keys in the dictionary yet
                 data[key] = perform_dict[key]

    # convert to dataframe
    behave_df = pd.DataFrame.from_dict(data)
    behave_df=behave_df.rename(columns={'sensitivity ': 'sensitivity'})

    # reject subjects who aren't responding enough
    behave_df = behave_df[~behave_df['rejectsubject']]

    # get inverse efficiency
    behave_df["InverseEfficiency"] = behave_df["RT"] / (behave_df["correct"] / 100)

    # plot results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.swarmplot("Testday", y="RT", data=behave_df, ax=ax1, color=".2")
    sns.violinplot(x="Testday", y="RT",  data=behave_df,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, )

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim([0.4, 1.2])

    titlestring = 'Motion Task RT by Day Train Cue Mean ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot results - standard deviation
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.swarmplot("Testday", y="RTSTD", data=behave_df, ax=ax1, color=".2")
    sns.violinplot(x="Testday", y="RTSTD",  data=behave_df,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([0.4, 1.2])

    titlestring = 'Motion Task RT Std Dev by Day Train Cue Mean' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot results - inverse efficiency
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.swarmplot("Testday", y="InverseEfficiency", data=behave_df, ax=ax1, color=".2")
    sns.violinplot(x="Testday", y="InverseEfficiency",  data=behave_df,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, )

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([0.4, 1.2])

    titlestring = 'Motion Task RT Inverse Efficiency by Day Train Cue Mean ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    # plot results - a bunch of things
    plots = ["correct", "falsealarm", "criterion", "sensitivity"]

    fig, (ax1) = plt.subplots(2, 2, figsize=(10, 10))
    ax1 = ax1.flatten()
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    for ii in np.arange(4):
        sns.swarmplot(x="Testday", y=plots[ii], data=behave_df, ax=ax1[ii], color=".2")
        sns.violinplot(x="Testday", y=plots[ii],  data=behave_df,
                    palette=sns.color_palette(colors), style="ticks", ax=ax1[ii])

        ax1[ii].spines['top'].set_visible(False)
        ax1[ii].spines['right'].set_visible(False)

        ax1[ii].set_title(plots[ii])

        # ax1[ii].set_ylim([0, 100])
    titlestring = 'Motion Task Accuracy by Day Train Cue Mean' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')























def collate_behaviour_prepost_compare(settings):

    # get task specific settings
    data = {}
    settings = settings.get_settings_behave_prepost()

    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):  # cycle trough space, feature and sham train groups
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_behave_prepost()

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            filename =  bids.direct_results / Path(bids.substring + "motiondiscrim_cuemean_" + settings.string_testtrain[0] + ".npy")
            perform_dict = np.load(filename,allow_pickle=True)[()]

            #   Append to list
            for key in perform_dict.keys():
                if not(isinstance(perform_dict[key], (list))):
                    perform_dict[key] = perform_dict[key].tolist()
                try:                
                    data[key].extend(perform_dict[key])
                except KeyError: # For the first subject, we don't have any keys in the dictionary yet
                    data[key] = perform_dict[key]

    # convert to dataframe
    behave_df = pd.DataFrame.from_dict(data)
    behave_df=behave_df.rename(columns={'sensitivity ': 'sensitivity'})

    # get inverse efficiency
    behave_df["InverseEfficiency"] = behave_df["RT"] / (behave_df["correct"] / 100)

    # get results of classification
    df_classifier, df_classifier_condensed = datacompare.load_classifierdata(settings)

    # merge dataframes
    df_behaveresults = pd.merge( df_classifier_condensed[['SubID', 'ClassifierAccuracy']], behave_df)

    # reject subjects who aren't responding enough
    behave_df.loc[behave_df.correct<25, 'rejectsubject']=True
    
    # behave_df.loc[behave_df['SubID'].isin(['Space118', 'Space94']),'rejectsubject']=True # incredibly high false alarm rates
    toreject = behave_df[behave_df['rejectsubject']]['SubID']
    df_behaveresults = df_behaveresults[~df_behaveresults['SubID'].isin(toreject)]



    # Calculate training effects
    factors = ['ClassifierAccuracy', 'AttentionTrained']
    measures_diff = ['correct', 'incorrect', 'miss', 'falsealarm', 'sensitivity', 'criterion', 'LR',
       'RT', 'RTSTD', 'InverseEfficiency']
    
    tmp = df_behaveresults.pivot(columns='Testday', index='SubID').T.swaplevel().T #pivot

    df_behavetrain = tmp["Day 4"][measures_diff] - tmp["Day 1"][measures_diff] #Calculate training effects
    for factor in factors:
        df_behavetrain[factor] = tmp["Day 1"][factor]

    # subtract subject means
    behave_dfcorrected = behave_df.copy()
    behave_dfcorrected=behave_dfcorrected.set_index('SubID')
    means=(behave_dfcorrected.loc[behave_dfcorrected["Testday"]=="Day 1", measures_diff] + behave_dfcorrected.loc[behave_dfcorrected["Testday"]=="Day 4", measures_diff])/2
    for day in ["Day 1", "Day 4"]:
        dat = behave_dfcorrected.loc[behave_dfcorrected["Testday"]==day, measures_diff]
        behave_dfcorrected.loc[behave_dfcorrected["Testday"]==day, measures_diff] = dat[measures_diff] - means[measures_diff]


    # export results
    # # lets run some stats with R - save it out
    df_behaveresults.to_csv(bids.direct_results_group_compare / Path("motiondiscrim_behaveresults_CueMean.csv"),index=False)
    df_behavetrain.to_csv(bids.direct_results_group_compare / Path("motiondiscrim_behavetrainresults_CueMean.csv"),index=False)


    # simple stats
    import scipy.stats as stats

    datplot = df_behavetrain
    space = datplot[datplot['AttentionTrained'] == 'Space']
    feat = datplot[datplot['AttentionTrained'] == 'Feature']
    cat12 = datplot[datplot['AttentionTrained'].isin(['Feature', 'Space'])]
    sham = datplot[datplot['AttentionTrained'] == 'Sham']

    print(stats.ttest_ind(space['sensitivity'], sham['sensitivity']))
    print(stats.ttest_ind(feat['sensitivity'], sham['sensitivity']))
    print(stats.ttest_ind(cat12['sensitivity'], sham['sensitivity']))

    print(stats.ttest_1samp(space['sensitivity'], 0))
    print(stats.ttest_1samp(feat['sensitivity'], 0))
    print(stats.ttest_1samp(sham['sensitivity'], 0))

    print(df_behavetrain.groupby('AttentionTrained').count()['sensitivity'])

    # plot results
    ##########################################  plot day 1 Vs. Day 4 Results by task, and group ##########################################
    coloruse = [settings.lightteal, settings.medteal]
    datplot = behave_dfcorrected.copy()
    import analyse_motiontask as mottaskanal
    mottaskanal.plotgroupedresult_complex(datplot, "sensitivity", '', bids, coloruse, [-1, 1])
    mottaskanal.plotgroupedresult_complexNF(datplot, "sensitivity", '', bids, coloruse, [-1, 1])
    
    ##########################################  plottraining effects ##########################################
    datplot = pd.DataFrame(df_behavetrain.reset_index().to_dict())
    mottaskanal.plotbehavetrainingeffects_combined(datplot,  "sensitivity", 'Motion Task Sensitivity training effect combined cuemean', bids, settings, [-1, 1])
