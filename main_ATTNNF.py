# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import h5py
import matplotlib.pyplot as plt
import Analysis_Code.helperfunctions_ATTNNF as helper
import Analysis_Code.functions_getEEGprepost as geegpp
import Analysis_Code.functions_getEEG_duringNF as geegdnf
import Analysis_Code.analyse_visualsearchtask as avissearch
import Analysis_Code.analyse_nbacktask as anback

# P86 super artifacty throughout - maybe exclude? frequency spectrum looks weird. look into this.

# TODO:

#### New Analyses
# correlate behaviour with ssvep selectivity
# look at differences between classifiable and unclassifiable participants.
# analyse feedback - how long in each state? how does it correspond to behaviour?
# analyse behaviour during neurofeedback training - across the three days.
# look at withing session learning curves for SSVEPs
# analyse wavelets around movement epochs
# GLM of results
# Bayesian anova of results.

## Changes to Code
# asthetic changes to plotting - add subject scatterpoints to violin plots
# stats on behaviour
# integrate behavioural analyses with python
# group average topoplots

######## Decide which single subject analyses to do ########

analyse_behaviour_prepost = True
# analyse_behaviour_prepost = False # Analyse Behaviour Pre Vs. Post Training

# analyse_EEG_prepost =True # analyse EEG Pre Vs. Post Training
analyse_EEG_prepost =False # analyse EEG Pre Vs. Post Training

# analyse_EEG_duringNF = True # analyse EEG during Neurofeedback
analyse_EEG_duringNF = False # analyse EEG during Neurofeedback

# analyse_visualsearchtask = True # Analyse Visual Search Task
analyse_visualsearchtask = False # Analyse Visual Search Task

# analyse_nbacktask = True # Analyse N-back Task
analyse_nbacktask = False # Analyse N-back Task


######## Decide which group analyses to do ########

# collateEEGprepost = True# Collate EEG Pre Vs. Post Training across subjects
collateEEGprepost = False# Collate EEG Pre Vs. Post Training across subjects

# collateEEGprepostcompare = True # Collate EEG Pre Vs. Post Training across subjects
collateEEGprepostcompare = False # Collate EEG Pre Vs. Post Training across subjects

# collateEEG_duringNF = True # Collate EEG during Neurofeedback
collateEEG_duringNF = False # Collate EEG during Neurofeedback

# collate_visualsearchtask = True# Collate Visual Search results
collate_visualsearchtask = False # Collate Visual Search results

# collate_nbacktask = True # Analyse N-back Task
collate_nbacktask = False # Analyse N-back Task





# setup generic settings
attntrained = 0 # ["Space", "Feature"]
settings = helper.SetupMetaData(attntrained)

print("Analysing Data for condition train: " + settings.string_attntrained[settings.attntrained])

# iterate through subjects for individual subject analyses
for sub_count, sub_val in enumerate(settings.subsIDX):
    if (analyse_behaviour_prepost):
        # get task specific settings
        settings = settings.get_settings_behave_prepost()

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

            file2useIDX = np.argmax(filesizes)  # get the biggest file (there are often smaller shorter accidental recordings)
            file2use = possiblefiles[file2useIDX]

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
            num_responses = responses.shape[0]
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
                trialresponses = np.array(np.where(responses[:,1]==TT))
                trialresponses_accounted = np.zeros(len(trialresponses.T))

                # Allocate response accuracy and reaction time for this trial
                for ii in np.arange(len(correct)):
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
                        if responses[idx_response, 0] == correct[ii]: # Correct Response!
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_correct))
                        else:
                            resp_accuracy = np.row_stack((resp_accuracy, settings.responseopts_incorrect))

                        tmp = responsetime[TT, responses[idx_response,2]] - moveonsets[np.reshape(correctmoveorder, -1)[ii],TT]/settings.mon_ref
                        resp_reactiontime = np.row_stack((resp_reactiontime, tmp))
                        resp_trialattntype = np.row_stack((resp_trialattntype, trialattntype[TT]))

                    elif len(idx_response) > 1:  # False alarm somewhere
                        idx_correct = np.where(np.squeeze(responses[idx_response, 0] == correct[ii]))
                        idx_falsealarm = np.where(np.squeeze(responses[idx_response, 0] != correct[ii]))

                        # Accuracy
                        tmp = np.ones(idx_response.shape[1])*np.nan # create array of NaNs of length of number of responses
                        tmp[ idx_correct ] = settings.responseopts_correct # mark correct answers as correct
                        tmp[ idx_falsealarm ] = settings.responseopts_falsealarm  # mark incorrect answers as false alarms

                        resp_accuracy = np.row_stack((resp_accuracy, np.expand_dims(tmp, axis = 1)) )# stack to accuracy results

                        # Response Time and Trial Attention type
                        tmp = np.ones(idx_response.shape[1]) * np.nan  # create array of NaNs of length of number of responses
                        tmp[idx_correct] = responsetime[TT, responses[np.squeeze(idx_response)[idx_correct],2]] - moveonsets[np.reshape(correctmoveorder, -1)[ii],TT]/settings.mon_ref

                        resp_reactiontime = np.row_stack((resp_reactiontime,  np.expand_dims(tmp, axis = 1)))

                        # trial attention type
                        tmp = np.ones(idx_response.shape[1]) * trialattntype[TT]
                        resp_trialattntype = np.row_stack((resp_trialattntype,  np.expand_dims(tmp, axis = 1)))






    if (analyse_EEG_prepost):
        geegpp.analyseEEGprepost(settings, sub_val)

    if (analyse_EEG_duringNF):
        geegdnf.analyseEEG_duringNF(settings, sub_val)

    if (analyse_visualsearchtask):
        if (attntrained == 1):  # correct for lost data for sub 21 (feature train)
            if (attntrained == 1):  # correct for lost data for sub 21 (feature train)
                settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                    np.isin(settings.subsIDXcollate, np.array([21, 89])))
                settings.num_subs = settings.num_subs - 2
            if (attntrained == 0):  # correct for lost data for sub 21 (feature train)
                settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                    np.isin(settings.subsIDXcollate, np.array([90])))
                settings.num_subs = settings.num_subs - 1

        avissearch.analyse_visualsearchtask(settings, sub_val)

    if (analyse_nbacktask):
        if (attntrained == 1):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate, np.isin(settings.subsIDXcollate, np.array([21, 89 ])))
            settings.num_subs = settings.num_subs - 2
        if (attntrained == 0):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate, np.isin(settings.subsIDXcollate, np.array([90])))
            settings.num_subs = settings.num_subs - 1

        anback.analyse_nbacktask(settings, sub_val)


# Collate EEG prepost
if (collateEEGprepost):

    print('collating SSVEP amplitudes pre Vs. post training')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_prepost()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_prepost_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    fftdat_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    wavelets_prepost_group = np.empty((len(timepoints_zp) + 1, settings.num_days, settings.num_attd_unattd, settings.num_attnstates, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_pre_post_results.npz"), allow_pickle=True) #
        #saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

        # store results
        SSVEPs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean']
        SSVEPs_epochs_prepost_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']
        fftdat_group[:,:,:,:,:,sub_count] = results['fftdat']
        fftdat_epochs_group[:, :, :, :, :, sub_count] = results['fftdat_epochs']
        wavelets_prepost_group[:,:,:,:,sub_count] = results['wavelets_prepost']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

    np.savez(bids.direct_results_group / Path("EEGResults_prepost"),
             SSVEPs_prepost_group=SSVEPs_prepost_group,
             SSVEPs_epochs_prepost_group=SSVEPs_epochs_prepost_group,
             fftdat_group=fftdat_group,
             fftdat_epochs_group=fftdat_epochs_group,
             wavelets_prepost_group=wavelets_prepost_group,
             timepoints_use=timepoints_use,
             freq=freq)


    # plot grand average frequency spectrum
    fftdat_ave = np.mean(fftdat_group, axis = 5)
    geegpp.plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring='ERP', settings=settings, freq=freq)

    fftdat_epochs_ave = np.mean(fftdat_epochs_group, axis = 5)
    geegpp.plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    geegpp.plotGroupSSVEPsprepost(SSVEPs_prepost_group, bids, ERPstring='ERP', settings=settings)
    geegpp.plotGroupSSVEPsprepost(SSVEPs_epochs_prepost_group, bids, ERPstring='Single Trial', settings=settings)

    # plot wavelet results
    wavelets_prepost_ave = np.mean(wavelets_prepost_group, axis=4)
    wavelets_prepost_std = np.std(wavelets_prepost_group, axis=4)/num_subs

    # plot wavelet data
    fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(10, 15))
    for attn in np.arange(settings.num_attnstates):
        for dayuse in np.arange(settings.num_days):
            if (dayuse == 0): axuse = ax1[attn]
            if (dayuse == 1): axuse = ax2[attn]
            if (attn == 0):
                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 0, attn] - wavelets_prepost_std[:, dayuse, 0, attn],
                                   wavelets_prepost_ave[:, dayuse, 0, attn] + wavelets_prepost_std[:, dayuse, 0, attn], alpha=0.3, facecolor=settings.medteal)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 0, attn], color=settings.medteal,
                           label=settings.string_attd_unattd[0])

                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 1, attn] - wavelets_prepost_std[:, dayuse, 1, attn],
                                   wavelets_prepost_ave[:, dayuse, 1, attn] + wavelets_prepost_std[:, dayuse, 1, attn], alpha=0.3, facecolor=settings.medteal)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 1, attn], color=settings.lightteal,
                           label=settings.string_attd_unattd[1])
            else:
                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 0, attn] - wavelets_prepost_std[:, dayuse, 0, attn],
                                   wavelets_prepost_ave[:, dayuse, 0, attn] + wavelets_prepost_std[:, dayuse, 0, attn],
                                   alpha=0.3, facecolor=settings.orange)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 0, attn], color=settings.orange,
                           label=settings.string_attd_unattd[0])

                axuse.fill_between(timepoints_use,
                                   wavelets_prepost_ave[:, dayuse, 1, attn] - wavelets_prepost_std[:, dayuse, 1, attn],
                                   wavelets_prepost_ave[:, dayuse, 1, attn] + wavelets_prepost_std[:, dayuse, 1, attn],
                                   alpha=0.3, facecolor=settings.yellow)
                axuse.plot(timepoints_use, wavelets_prepost_ave[:, dayuse, 1, attn], color=settings.yellow,
                           label=settings.string_attd_unattd[1])
            axuse.set_xlim(-1, 6)
            axuse.set_xlabel('Time (s)')
            axuse.set_ylabel('MCA')
            axuse.set_ylim(100, 1000)
            axuse.legend()
            axuse.set_title(settings.string_cuetype[attn] + ' ' + settings.string_prepost[dayuse])

    titlestring = 'Group Mean wavelets pre-post ' + settings.string_attntrained[settings.attntrained]
    fig.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # save attentional selectivity for stats - single trial
    ssvep_selectivity_prepost = SSVEPs_epochs_prepost_group[0, :, :, :] - SSVEPs_epochs_prepost_group[1, :, :, :]
    tmp = np.reshape(ssvep_selectivity_prepost, (4,settings.num_subs)) # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost_epochs.npy"), tmp)

    # save attentional selectivity for stats
    ssvep_selectivity_prepost = SSVEPs_prepost_group[0, :, :, :] - SSVEPs_prepost_group[1, :, :, :]
    tmp = np.reshape(ssvep_selectivity_prepost,
                     (4, settings.num_subs))  # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
    np.save(bids.direct_results_group / Path("group_ssvep_selectivity_prepost.npy"), tmp)

# Collate EEG prepost - Compare space and feature attention
if (collateEEGprepostcompare):
    import seaborn as sns
    import pandas as pd
    print('collating SSVEP amplitudes pre Vs. post training compareing Space Vs. Feat Training')

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
        settings = settings.get_settings_nbacktask()

        # file names
        bids = helper.BIDS_FileNaming(0, settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results_group / Path("EEGResults_prepost.npz"), allow_pickle=True)  #

        SSVEPs_epochs_prepost_group = results['SSVEPs_epochs_prepost_group'] #results['SSVEPs_prepost_group']
        diffdat = SSVEPs_epochs_prepost_group[0, :, :, :] - SSVEPs_epochs_prepost_group[1, :, :, :] # [day,attn,sub]

        # store results for attention condition
        tmp = [settings.string_prepost[0]] * settings.num_subs * settings.num_attnstates + [settings.string_prepost[1]] * settings.num_subs * settings.num_attnstates
        daystrings = np.concatenate((daystrings, tmp))

        tmp = [settings.string_attntrained[0]] * settings.num_subs + [settings.string_attntrained[1]] * settings.num_subs
        attntaskstrings = np.concatenate((attntaskstrings, tmp, tmp))

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days * settings.num_attnstates
        attnstrings = np.concatenate((attnstrings, tmp))

        tmp = np.concatenate((diffdat[0, 0, :], diffdat[0, 1, :], diffdat[1, 0, :], diffdat[1, 1, :]))
        selectivity_compare = np.concatenate((selectivity_compare, tmp))

    data = {'Testday': daystrings, 'Attention Type': attntaskstrings, 'Attention Trained': attnstrings, 'Selectivity (ΔµV)':  selectivity_compare}
    df_selctivity = pd.DataFrame(data)

    # plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = ["#F2B035", "#EC553A"]

    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y= "Selectivity (ΔµV)" , hue = "Testday", data=df_selctivity[df_selctivity["Attention Type"].isin([settings.string_attntrained[0]])], palette=sns.color_palette(colors), ax=ax1, split=True, inner="quartile")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title(settings.string_attntrained[0] + " Attention")
    ax1.set_ylim(-0.25, 0.65)
    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y="Selectivity (ΔµV)", hue="Testday",
                   data=df_selctivity[df_selctivity["Attention Type"].isin([settings.string_attntrained[1]])],
                   palette=sns.color_palette(colors), ax=ax2, split=True, inner="quartile")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title(settings.string_attntrained[1] + " Attention")
    ax2.set_ylim(-0.25, 0.65)


    titlestring = 'Attentional Selectivity PrePost Compare Training'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

# Collate EEG prepost
if (collateEEG_duringNF):
    print('collating SSVEP amplitudes during NF')
    # get settings specific to this analysis
    settings = settings.get_settings_EEG_duringNF()

    # Get timing settings
    timelimits_data_zp, timepoints_zp, frequencypoints_zp, zeropoint_zp = helper.get_timing_variables(settings.timelimits_zeropad, settings.samplingfreq)

    # preallocate group mean variables
    num_subs = settings.num_subs
    SSVEPs_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    SSVEPs_epochs_group = np.empty((settings.num_attd_unattd, settings.num_days, settings.num_attnstates, num_subs))
    fftdat_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))
    fftdat_epochs_group = np.empty((settings.num_electrodes, len(timepoints_zp) + 1, settings.num_attnstates, settings.num_levels, settings.num_days, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "EEG_duringNF_results.npz"), allow_pickle=True) #
        #saved vars: SSVEPs_prepost_channelmean, SSVEPs_prepost_channelmean_epochs, wavelets_prepost, timepoints_zp, erps_days_wave, fftdat, fftdat_epochs, freq)

        # store results
        SSVEPs_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean']
        SSVEPs_epochs_group[:, :, :, sub_count] = results['SSVEPs_prepost_channelmean_epochs']
        fftdat_group[:,:,:,:,:,sub_count] = results['fftdat']
        fftdat_epochs_group[:, :, :, :, :, sub_count] = results['fftdat_epochs']

        timepoints_use = results['timepoints_zp']
        freq = results['freq']

    # plot grand average frequency spectrum
    fftdat_ave = np.nanmean(fftdat_group, axis = 5)
    geegdnf.plotGroupFFTSpectrum(fftdat_ave, bids, ERPstring='ERP', settings=settings, freq=freq)

    fftdat_epochs_ave = np.nanmean(fftdat_epochs_group, axis = 5)
    geegdnf.plotGroupFFTSpectrum(fftdat_epochs_ave, bids, ERPstring='Single Trial', settings=settings, freq=freq)

    # plot average SSVEP results
    geegdnf.plotGroupSSVEPs(SSVEPs_group, bids, ERPstring='ERP', settings=settings)
    geegdnf.plotGroupSSVEPs(SSVEPs_epochs_group, bids, ERPstring='Single Trial', settings=settings)

# Collate Visual Search Task
if (collate_visualsearchtask):
    print('Collating Visual Search Task')

    # get task specific settings
    settings = settings.get_settings_visualsearchtask()

    # correct for lost data
    if (attntrained == 1):  # correct for lost data for sub 21 (feature train)
        settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                            np.isin(settings.subsIDXcollate, np.array([21, 89])))
        settings.num_subs = settings.num_subs - 2
    if (attntrained == 0):  # correct for lost data for sub 21 (feature train)
        settings.subsIDXcollate = np.delete(settings.subsIDXcollate, np.isin(settings.subsIDXcollate, np.array([90])))
        settings.num_subs = settings.num_subs - 1

    # preallocate group mean variables
    num_subs = settings.num_subs
    acc_vissearch_all = np.empty((settings.num_trialscond, settings.num_setsizes, settings.num_days, num_subs))
    rt_vissearch_all = np.empty((settings.num_trialscond, settings.num_setsizes, settings.num_days, num_subs))
    mean_acc_all = np.empty((settings.num_setsizes, settings.num_days, num_subs))
    mean_rt_all = np.empty((settings.num_setsizes, settings.num_days, num_subs))

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        results = np.load(bids.direct_results / Path(bids.substring + "visual_search_results.npz"), allow_pickle=True)  #
        # saved vars: meanacc=meanacc, meanrt=meanrt, acc_vissearch=acc_vissearch, rt_vissearch=rt_vissearch

        # store results
        acc_vissearch_all[ :, :, :, sub_count] = results['acc_vissearch']
        mean_acc_all[ :, :, sub_count] = results['meanacc']

        rt_vissearch_all[ :, :, :, sub_count] = results['rt_vissearch']
        mean_rt_all[ :, :, sub_count] = results['meanrt']


    # plot accuracy results
    meanacc = np.nanmean(mean_acc_all, axis=2)
    erroracc = np.empty((settings.num_setsizes, settings.num_days))
    erroracc[:, 0] = helper.within_subjects_error(np.squeeze(mean_acc_all[:,0,:]).T)
    erroracc[:, 1] = helper.within_subjects_error(np.squeeze(mean_acc_all[:,1,:]).T)

    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_prepost
    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, meanacc[0, :], width, yerr=erroracc[0, :], label=settings.string_setsize[0], facecolor=settings.lightteal)
    plt.bar(x, meanacc[1, :], width,  yerr=erroracc[1, :],label=settings.string_setsize[1], facecolor=settings.medteal)
    plt.bar(x + width, meanacc[2, :], width,yerr=erroracc[2, :],  label=settings.string_setsize[2], facecolor=settings.darkteal)

    plt.ylim([90, 100])
    plt.ylabel('Accuracy (%)')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = 'Visual Search Accuracy train ' + settings.string_attntrained[settings.attntrained]
    plt.title(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    # plot reaction time results
    meanrt = np.nanmean(mean_rt_all, axis=2)
    errorrt = np.empty((settings.num_setsizes, settings.num_days))
    errorrt[:, 0] = helper.within_subjects_error(np.squeeze(mean_rt_all[:,0,:]).T)
    errorrt[:, 1] = helper.within_subjects_error(np.squeeze(mean_rt_all[:,1,:]).T)

    fig, ax = plt.subplots(figsize=(5, 5))

    labels = settings.string_prepost
    x = np.arange(len(labels))
    width = 0.25

    plt.bar(x - width, meanrt[0, :], width, yerr=errorrt[0, :], label=settings.string_setsize[0], facecolor=settings.lightteal)
    plt.bar(x, meanrt[1, :], width,  yerr=errorrt[1, :],label=settings.string_setsize[1], facecolor=settings.medteal)
    plt.bar(x + width, meanrt[2, :], width,yerr=errorrt[2, :],  label=settings.string_setsize[2], facecolor=settings.darkteal)

    plt.ylabel('reaction time (s)')
    plt.xticks(x, labels)
    plt.legend()
    ax.set_frame_on(False)

    titlestring = 'Visual Search reaction time train ' + settings.string_attntrained[settings.attntrained]
    plt.title(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')









    # Collate across group
    import seaborn as sns
    import pandas as pd

    # preallocate
    num_subs = np.zeros((settings.num_attnstates))
    daystrings = []
    attnstrings = []
    setsizestrings = []
    # substring = []
    accuracy_compare = []
    rt_compare = []

    meandaystrings = []
    meanattnstrings = []
    meanaccuracy_compare = []
    meanrt_compare = []

    print('Collating Visual Search Task for space and feature train')
    for attntrained in np.arange(settings.num_attnstates):  # cycle trough space and feature train groups

        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_visualsearchtask()

        # correct for lost data
        if (attntrained == 1):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([21, 89])))
            settings.num_subs = settings.num_subs - 2
        if (attntrained == 0):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([90])))
            settings.num_subs = settings.num_subs - 1

        num_subs[attntrained] = settings.num_subs
        mean_acc_all = np.empty((settings.num_setsizes, settings.num_days, settings.num_subs))
        mean_rt_all = np.empty((settings.num_setsizes, settings.num_days, settings.num_subs))

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "visual_search_results.npz"),
                              allow_pickle=True)  # saved vars: meanacc=meanacc, meanrt=meanrt, acc_vissearch=acc_nback, rt_vissearch=rt_nback

            # store results temporarily
            mean_acc_all[:,: , sub_count] = results['meanacc']
            mean_rt_all[:,:, sub_count] = results['meanrt']

        # store results for attention condition
        tmp = [settings.string_prepost[0]] * (settings.num_subs * settings.num_setsizes) + [settings.string_prepost[1]] * (settings.num_subs * settings.num_setsizes)
        daystrings = np.concatenate((daystrings, tmp)) # pretrain then postrain

        tmp = [settings.string_setsize[0]] * (settings.num_subs) +  [settings.string_setsize[1]] * (settings.num_subs) +  [settings.string_setsize[2]] * (settings.num_subs)
        setsizestrings = np.concatenate((setsizestrings, tmp, tmp)) #Each setsize for each subject, repeated for the two testdays

        # tmp = np.arange(settings.num_subs)
        # substring = np.concatenate((substring, tmp, tmp, tmp, tmp, tmp, tmp)) # subject number for each setsize and day cond

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days * settings.num_setsizes
        attnstrings = np.concatenate((attnstrings, tmp)) # All setsizes and subjects and days are the same attention trained

        tmp = np.concatenate((mean_acc_all[0, 0, :], mean_acc_all[1, 0, :], mean_acc_all[2, 0, :],
                              mean_acc_all[0, 1, :], mean_acc_all[1, 1, :], mean_acc_all[2, 1, :]))
        accuracy_compare = np.concatenate((accuracy_compare, tmp))

        tmp = np.concatenate((mean_rt_all[0, 0, :], mean_rt_all[1, 0, :], mean_rt_all[2, 0, :],
                              mean_rt_all[0, 1, :], mean_rt_all[1, 1, :], mean_rt_all[2, 1, :]))
        rt_compare = np.concatenate((rt_compare, tmp))

        # store results for attention condition - mean across set size conditions
        tmp = [settings.string_prepost[0]] * settings.num_subs + [settings.string_prepost[1]] * settings.num_subs
        meandaystrings = np.concatenate((meandaystrings, tmp))

        tmp = [settings.string_attntrained[attntrained]] * settings.num_subs * settings.num_days
        meanattnstrings = np.concatenate((meanattnstrings, tmp))

        mean_acc_all2 = np.mean(mean_acc_all, axis = 0)
        tmp = np.concatenate((mean_acc_all2[0, :], mean_acc_all2[1, :]))
        meanaccuracy_compare = np.concatenate((meanaccuracy_compare, tmp))

        mean_rt_all2 = np.mean(mean_rt_all, axis = 0)
        tmp = np.concatenate((mean_rt_all2[0, :], mean_rt_all2[1, :]))
        meanrt_compare = np.concatenate((meanrt_compare, tmp))

    # create the data frames for accuracy and reaction time data
    data = {'Testday': daystrings, 'Attention Trained': attnstrings, 'Set Size':setsizestrings, 'Accuracy (%)': accuracy_compare}
    df_acc = pd.DataFrame(data)

    data = { 'Testday': daystrings, 'Attention Trained': attnstrings, 'Set Size':setsizestrings, 'Reaction Time (s)': rt_compare}
    df_rt = pd.DataFrame(data)

    # plot results
    fig, (ax1) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    for attn, attnstring in enumerate(settings.string_attntrained):
        sns.violinplot(x="Set Size", y="Reaction Time (s)", hue="Testday",
                       data=df_rt[df_rt["Attention Trained"].isin([attnstring])],
                       palette=sns.color_palette(colors), style="ticks", ax=ax1[attn], split=True, inner="stick")

        ax1[attn].spines['top'].set_visible(False)
        ax1[attn].spines['right'].set_visible(False)

        ax1[attn].set_title(attnstring)

    titlestring = 'Visual Search Results Compare Training'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')



    # Plot average over set sizes
    # create the data frames for accuracy and reaction time data
    data = {'Testday': meandaystrings, 'Attention Trained': meanattnstrings, 'Accuracy (%)': meanaccuracy_compare}
    df_acc_SSmean = pd.DataFrame(data)

    data = {'Testday': meandaystrings, 'Attention Trained': meanattnstrings, 'Reaction Time (s)': meanrt_compare}
    df_rt_SSmean = pd.DataFrame(data)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = ["#112F41", "#4CB99F"]

    # Accuracy Grouped violinplot

    sns.violinplot(x="Attention Trained", y= "Accuracy (%)" , hue = "Testday", data=df_acc_SSmean, palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 100)
    ax1.set_title("Accuracy")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]
    sns.violinplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", data=df_rt_SSmean,
                   palette=sns.color_palette(colors), style="ticks", ax=ax2, split=True, inner="stick")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title("Reaction time")

    titlestring = 'Visual Search Results Compare Training Set Size Ave'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

# Collate N-Back Task
if (collate_nbacktask):
    import seaborn as sns
    import pandas as pd

    # preallocate
    num_subs = np.zeros((settings.num_attnstates))
    daystrings = []
    attnstrings = []
    accuracy_compare = []
    rt_compare = []

    print('Collating N-back Task for space and feature train')

    for attntrained in np.arange(settings.num_attnstates): # cycle trough space and feature train groups

        # get task specific settings
        settings = helper.SetupMetaData(attntrained)
        settings = settings.get_settings_nbacktask()

        # pre-allocate for this group
        if (attntrained == 1):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([21, 89])))
            settings.num_subs = settings.num_subs - 2
        if (attntrained == 0):  # correct for lost data for sub 21 (feature train)
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                np.isin(settings.subsIDXcollate, np.array([90])))
            settings.num_subs = settings.num_subs - 1

        num_subs[attntrained] = settings.num_subs
        mean_acc_all = np.empty((settings.num_days, settings.num_subs))
        mean_rt_all = np.empty((settings.num_days, settings.num_subs))

        # iterate through subjects for individual subject analyses
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):

            # get directories and file names
            bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
            print(bids.substring)

            # load results
            results = np.load(bids.direct_results / Path(bids.substring + "Nback_results.npz"),
                              allow_pickle=True)  #saved vars: meanacc=meanacc, meanrt=meanrt, acc_nback=acc_nback, rt_nback=rt_nback

            # store results temporarily
            mean_acc_all[:, sub_count] = results['meanacc']*100
            mean_rt_all[:, sub_count] = results['meanrt']

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
    data = {'Testday': daystrings, 'Attention Trained': attnstrings, 'Accuracy (%)': accuracy_compare }
    df_acc = pd.DataFrame(data)

    data = {'Testday': daystrings, 'Attention Trained': attnstrings, 'Reaction Time (s)': rt_compare}
    df_rt = pd.DataFrame(data)

    # plot results

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = ["#112F41", "#4CB99F"]

    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y= "Accuracy (%)" , hue = "Testday", data=df_acc, palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim(0, 100)
    ax1.set_title("Accuracy")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]
    sns.violinplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", data=df_rt,
                   palette=sns.color_palette(colors), style="ticks", ax=ax2, split=True, inner="stick")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title("Reaction time")

    titlestring = 'Nback Results Compare Training'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

