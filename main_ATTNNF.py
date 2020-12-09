# Import nescescary packages
import numpy as np
from pathlib import Path
import mne
import h5py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys

import Analysis_Code.helperfunctions_ATTNNF as helper
import Analysis_Code.functions_getEEGprepost as geegpp
import Analysis_Code.functions_getEEG_duringNF as geegdnf
import Analysis_Code.analyse_visualsearchtask as avissearch
import Analysis_Code.analyse_nbacktask as anback
import Analysis_Code.analyse_motiontask_prepost as analyse_motion_prepost

# dispay all rows
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 10)

# P86 super artifacty throughout - exlude? - frequency spectrum looks weird. look into this. - can we exclude a specific channel?
# P106 - exclude day 1 phase train
# exclude P118 from everything for horrifically bad performance.
# exclude chance level vis search Ps
# figure out whats happening with visual search collation.

# TODO:

#### New Analyses
# analyse the neurofeedback - how much time are we spending in each state and does that actually have an effect
# analyse the latency - are we changing the amount of time people spend in each state? (sustained attn feedback)
# variance in results

# correlate behaviour with ssvep selectivity
# look at differences between classifiable and unclassifiable participants.
# analyse feedback - how long in each state? how does it correspond to behaviour?
# look at withing session learning curves for SSVEPs
# analyse wavelets around movement epochs
# GLM of results
# Bayesian anova of results.
# correlate attn and WM scores

## Changes to Code
# stats on behaviour
# group average topoplots
# individual trial independance of spatial and feature based-attention measure - how correlated are they really?

######## Decide which single subject analyses to do ########

# analyse_behaviour_prepost = True # Analyse Behaviour Pre Vs. Post Training
analyse_behaviour_prepost = False # Analyse Behaviour Pre Vs. Post Training

# analyse_behaviour_duringNF = True # Analyse Behaviour Pre Vs. Post Training
analyse_behaviour_duringNF = False # Analyse Behaviour Pre Vs. Post Training

# analyse_EEG_prepost =True # analyse EEG Pre Vs. Post Training
analyse_EEG_prepost =False # analyse EEG Pre Vs. Post Training
#
# analyse_EEG_duringNF = True # analyse EEG during Neurofeedback
analyse_EEG_duringNF = False # analyse EEG during Neurofeedback

# analyse_visualsearchtask = True # Analyse Visual Search Task
analyse_visualsearchtask = False # Analyse Visual Search Task

# analyse_nbacktask = True # Analyse N-back Task
analyse_nbacktask = False # Analyse N-back Task
#

######## Decide which group analyses to do ########

# collate_behaviour_prepost = True # Collate Behaviour Pre Vs. Post Training
collate_behaviour_prepost = False # Collate Behaviour Pre Vs. Post Training

# collate_behaviour_prepost_compare = True # Collate Behaviour Pre Vs. Post Training compare training groups
collate_behaviour_prepost_compare = False # Collate Behaviour Pre Vs. Post Training compare training groups

# collate_behaviour_duringNF = True # Collate Behaviour during Training
collate_behaviour_duringNF = False # Collate Behaviour during Training

# collate_behaviour_duringNF_compare = True # Collate Behaviourduring Training compare training groups
collate_behaviour_duringNF_compare = False # Collate Behaviour during Training compare training groups

# collateEEGprepost = True# Collate EEG Pre Vs. Post Training across subjects
collateEEGprepost = False# Collate EEG Pre Vs. Post Training across subjects
#
# collateEEGprepostcompare = True # Collate EEG Pre Vs. Post Training across subjects
collateEEGprepostcompare = False # Collate EEG Pre Vs. Post Training across subjects

# collateEEG_duringNF = True # Collate EEG during Neurofeedback
collateEEG_duringNF = False # Collate EEG during Neurofeedback
#
# collate_visualsearchtask = True # Collate Visual Search results
collate_visualsearchtask = False # Collate Visual Search results

# collate_nbacktask = True # Analyse N-back Task
collate_nbacktask = False # Analyse N-back Task

classification_acc_correlations = True # Assess whether classification accuracy correlated with training effects
# classification_acc_correlations = False # Assess whether classification accuracy correlated with training effects

# setup generic settings
attntrained = 0 # ["Space", "Feature"]
settings = helper.SetupMetaData(attntrained)

print("Analysing Data for condition train: " + settings.string_attntrained[settings.attntrained])

# iterate through subjects for individual subject analyses
for sub_count, sub_val in enumerate(settings.subsIDX):
    print(sub_val)
    if (analyse_behaviour_prepost):
        test_train = 0
        analyse_motion_prepost.run(settings, sub_val, test_train)

    if (analyse_behaviour_duringNF):
        test_train=1
        analyse_motion_prepost.run(settings, sub_val, test_train)

    if (analyse_EEG_prepost):
        geegpp.analyseEEGprepost(settings, sub_val)

    if (analyse_EEG_duringNF):
        geegdnf.analyseEEG_duringNF(settings, sub_val)

    if (analyse_visualsearchtask):
        if (attntrained == 1):  # correct for lost data
            if (attntrained == 1):  # correct for lost data for feature train
                settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                    np.isin(settings.subsIDXcollate, np.array([89])))
                settings.num_subs = settings.num_subs - 2

            if (attntrained == 0):  # correct for lost data for space train
                settings.subsIDXcollate = np.delete(settings.subsIDXcollate,
                                                    np.isin(settings.subsIDXcollate, np.array([90])))
                settings.num_subs = settings.num_subs - 1

        # Run analysis
        avissearch.analyse_visualsearchtask(settings, sub_val)

    if (analyse_nbacktask):
        if (attntrained == 1):  # correct for lost data for feature train
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate, np.isin(settings.subsIDXcollate, np.array([ 89 ])))
            settings.num_subs = settings.num_subs - 2

        if (attntrained == 0):  # correct for lost data for space train
            settings.subsIDXcollate = np.delete(settings.subsIDXcollate, np.isin(settings.subsIDXcollate, np.array([90])))
            settings.num_subs = settings.num_subs - 1

        # Run analysis
        anback.analyse_nbacktask(settings, sub_val)

# Collate motion task behaviour prepost
if (collate_behaviour_prepost):
    print('Collating Motion Task Behaviour')

    # get task specific settings
    settings = settings.get_settings_behave_prepost()

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):
        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        accdat_targ_sub = pd.read_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_acctarget_" +  settings.string_testtrain[0] + ".pkl"))
        accdat_dist_sub = pd.read_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_accdistract_" + settings.string_testtrain[0] + ".pkl"))
        behdat_sub = pd.read_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_allbehave_" +  settings.string_testtrain[0] + ".pkl"))

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
        accdat_targ_sub['subID']=sub_count
        accdat_dist_sub['subID'] = sub_count
        behdat_sub['subID']=sub_count

        accdat_targ_sub['subIDval'] = sub_val
        accdat_dist_sub['subIDval'] = sub_val
        behdat_sub['subIDval'] = sub_val

        # Stack across subjects
        if (sub_count==0): # First subject, no dataframe exists yet
            accdat_targ_all = accdat_targ_sub
            accdat_dist_all = accdat_dist_sub
            behdat_all = behdat_sub
        else:
            accdat_targ_all = accdat_targ_all.append(accdat_targ_sub, ignore_index=True) # ignore index just means the index will count all the way up
            accdat_dist_all = accdat_dist_all.append(accdat_dist_sub, ignore_index=True)  # ignore index just means the index will count all the way up
            behdat_all = behdat_all.append(behdat_sub, ignore_index=True)

    # average and plot reaction time data
    behdat_all_avg = behdat_all.groupby(["subID", "Testday", "Attention Type"]).mean()
    behdat_all_avg=behdat_all_avg.reset_index()

    # plot results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="Reaction Time", hue="Testday",data=behdat_all_avg , palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim([0.4, 1.2])

    titlestring =  'Motion Task RT by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # Plot Accuracy Data

     ############## Get Sensitivity
    from scipy.stats import norm
    import math

    Z = norm.ppf # percentile point function - normal distribution between 0 and 1.

    N_distractors = 336  # number of distractor events per day and condition.
    N_targets = 144  # number of target events per day and condition.


    dat = accdat_targ_all.loc[:, "correct"]/100 # hitrate

    dat[dat == 0] = 1/(2*N_targets) # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_targets)  # correct for zeros and ones (McMillan & Creelman, 2004)

    hitrate_zscore = Z(dat)

    dat = accdat_dist_all.loc[:, "falsealarm"]/100

    dat[dat == 0] = 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)
    dat[dat == 1] = 1 - 1 / (2 * N_distractors)  # correct for zeros and ones (McMillan & Creelman, 2004)

    falsealarmrate_zscore = Z(dat)

    accdat_targ_all.loc[:, "Sensitivity"] = hitrate_zscore - falsealarmrate_zscore
    accdat_targ_all.loc[:, "Criterion"] = 0.5 * (hitrate_zscore + falsealarmrate_zscore)
    accdat_targ_all.loc[:, "LikelihoodRatio"] = accdat_targ_all.loc[:, "Sensitivity"] * accdat_targ_all.loc[:, "Criterion"]

    # plot results - sensitivity
    plots = ["correct", "miss", "correctreject", "falsealarm"]

    fig, (ax1) = plt.subplots(2, 2, figsize=(10,10))
    ax1 = ax1.flatten()
    sns.set(style="ticks")
    colors = ["#F2B035", "#EC553A"]
    # Reaction time Grouped violinplot
    for ii in np.arange(4):
        if ii < 2:
            sns.violinplot(x="Attention Type", y=plots[ii], hue="Testday",data=accdat_targ_all , palette=sns.color_palette(colors), style="ticks", ax=ax1[ii], split=True, inner="stick")
        else:
            sns.violinplot(x="Attention Type", y=plots[ii], hue="Testday", data=accdat_dist_all, palette=sns.color_palette(colors), style="ticks", ax=ax1[ii], split=True, inner="stick")

        ax1[ii].spines['top'].set_visible(False)
        ax1[ii].spines['right'].set_visible(False)

        ax1[ii].set_title(plots[ii])

        ax1[ii].set_ylim([0, 100])
    titlestring =  'Motion Task Accuracy by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    # plot sensitivity results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="Sensitivity", hue="Testday",data=accdat_targ_all , palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_ylim([-1,5])

    titlestring =  'Motion Task Sensitivity by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    # plot Criterion results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="Criterion", hue="Testday",data=accdat_targ_all , palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([-1,5])

    titlestring =  'Motion Task Criterion by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')


    # plot Likelihood Ratio results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = ["#F2B035", "#EC553A"]

    sns.violinplot(x="Attention Type", y="LikelihoodRatio", hue="Testday",data=accdat_targ_all , palette=sns.color_palette(colors), style="ticks", ax=ax1, split=True, inner="stick")

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    # ax1.set_ylim([-1,5])

    titlestring =  'Motion Task Likelihood Ratio by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # Save results out for collation later
    accdat_targ_all['RT'] = behdat_all_avg['Reaction Time']
    accdat_targ_all.to_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))

if (collate_behaviour_prepost_compare):

    # get task specific settings
    settings = settings.get_settings_behave_prepost()

    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):  # cycle trough space and feature train groups
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_behave_prepost()

        # file names
        bids = helper.BIDS_FileNaming(subject_idx=0, settings=settings, day_val=0)
        print(bids.direct_results_group)
        # accdat_all_avg.to_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))
        df_behaveresults_tmp = pd.read_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))
        df_behaveresults_tmp['AttentionTrained'] = attntrained

        if (attntrainedcount == 0):
            df_behaveresults = df_behaveresults_tmp[['subID','subIDval', 'AttentionTrained', 'Attention Type', 'Testday', 'Sensitivity', 'Criterion', 'correct', 'LikelihoodRatio', 'RT']]
        else:
            df_behaveresults_tmp['subID'] = df_behaveresults_tmp['subID'] + 34
            df_behaveresults = df_behaveresults.append(df_behaveresults_tmp[['subID', 'subIDval', 'AttentionTrained', 'Attention Type', 'Testday', 'Sensitivity', 'Criterion', 'correct', 'LikelihoodRatio', 'RT']])

    # Exclude extremely poor performers.
    tmp = df_behaveresults.groupby('subIDval').mean()['correct']
    exclude = df_behaveresults.groupby('subIDval').mean()[tmp<10] # SUBIDs 1,2,23, 90

    df_behaveresults_clean = df_behaveresults[~df_behaveresults['subID'].isin(exclude['subID'])]

    # # lets run some stats with R - save it out
    df_behaveresults_clean.to_csv(bids.direct_results_group_compare / Path("motiondiscrim_behaveresults_ALL.csv"), index=False)


    ##########################################  plot day 1 Vs. Day 4 Results ##########################################
    #### RT data ####

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal]

    sns.swarmplot(x="Testday", y="RT", data=df_behaveresults_clean, color="0", alpha=0.3)
    sns.violinplot(x="Testday", y="RT", data=df_behaveresults_clean , palette=sns.color_palette(colors), style="ticks", ax=ax1, inner="box",  alpha=0.6)


    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring =  'Motion Task RT by Day pre Vs. post'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    #### Sensitivity data ####

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal]

    sns.swarmplot(x="Testday", y="Sensitivity", data=df_behaveresults_clean, color="0", alpha=0.3)
    sns.violinplot(x="Testday", y="Sensitivity", data=df_behaveresults_clean, palette=sns.color_palette(colors), style="ticks", ax=ax1, inner="box", alpha=0.6)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task Sensitivity by Day pre Vs. post'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    #### Correct Rate data ####

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal]

    sns.swarmplot(x="Testday", y="correct", data=df_behaveresults_clean, color="0", alpha=0.3)
    sns.violinplot(x="Testday", y="correct", data=df_behaveresults_clean, palette=sns.color_palette(colors), style="ticks", ax=ax1, inner="box", alpha=0.6)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task Correct Rate by Day pre Vs. post'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    #### Criterion data ####

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal]

    sns.swarmplot(x="Testday", y="Criterion", data=df_behaveresults_clean, color="0", alpha=0.3)
    sns.violinplot(x="Testday", y="Criterion", data=df_behaveresults_clean, palette=sns.color_palette(colors), style="ticks", ax=ax1, inner="box", alpha=0.6)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task Criterion by Day pre Vs. post'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    #### Likelihood Ratio ####

    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal]

    sns.swarmplot(x="Testday", y="LikelihoodRatio", data=df_behaveresults_clean, color="0", alpha=0.3)
    sns.violinplot(x="Testday", y="LikelihoodRatio", data=df_behaveresults_clean, palette=sns.color_palette(colors), style="ticks", ax=ax1, inner="box", alpha=0.6)

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task Likelihood Ratio by Day pre Vs. post'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    ##########################################  Calculate training effects  ##########################################
    idx_d1 = df_behaveresults_clean["Testday"]=="Day 1"
    idx_d4 = df_behaveresults_clean["Testday"] == "Day 4"

    tmpd4 = df_behaveresults_clean[idx_d4].reset_index()
    tmpd1 = df_behaveresults_clean[idx_d1].reset_index()

    df_behtraineffects = tmpd4[["subID", "AttentionTrained", "Attention Type"]].copy()

    df_behtraineffects["∆Sensitivity"] = tmpd4['Sensitivity'] - tmpd1['Sensitivity']
    df_behtraineffects["∆Criterion"] = tmpd4['Criterion'] - tmpd1['Criterion']
    df_behtraineffects["∆Correct"] = tmpd4['correct'] - tmpd1['correct']
    df_behtraineffects["∆RT"] = tmpd4['RT'] - tmpd1['RT']
    df_behtraineffects["∆LikelihoodRatio"] = tmpd4['LikelihoodRatio'] - tmpd1['LikelihoodRatio']

    # df_behtraineffects = df_behtraineffects.reset_index()
    ##########################################  plot training effects against attention trained and attention type ##########################################
    #### Reaction Time ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange]
    for i in np.arange(2):

        datplot = df_behtraineffects[df_behtraineffects ["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Type", y="∆RT", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Type", y="∆RT", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task RT training effect by attention'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    #### Sensitivity ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange]
    for i in np.arange(2):
        datplot = df_behtraineffects[df_behtraineffects["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Type", y="∆Sensitivity", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Type", y="∆Sensitivity", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task Sensitivity training effect by attention'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    #### Correct rate ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange]
    for i in np.arange(2):
        datplot = df_behtraineffects[df_behtraineffects["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Type", y="∆Correct", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Type", y="∆Correct", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task Correct training effect by attention'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    #### Criterion rate ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange]
    for i in np.arange(2):
        datplot = df_behtraineffects[df_behtraineffects["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Type", y="∆Criterion", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Type", y="∆Criterion", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task Criterion training effect by attention'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    #### Likelihood Ratio ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.yellow, settings.orange]
    for i in np.arange(2):
        datplot = df_behtraineffects[df_behtraineffects["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Attention Type", y="∆LikelihoodRatio", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Attention Type", y="∆LikelihoodRatio", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task Likelihood Ratio training effect by attention'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

# Collate motion task behaviour prepost
if (collate_behaviour_duringNF):
    print('Collating Motion Task Behaviour during Neurofeedback')

    # get task specific settings
    settings = settings.get_settings_behave_duringNF()

    # iterate through subjects for individual subject analyses
    for sub_count, sub_val in enumerate(settings.subsIDXcollate):

        # get directories and file names
        bids = helper.BIDS_FileNaming(int(sub_val), settings, 0)
        print(bids.substring)

        # load results
        accdat_targ_sub = pd.read_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_acctarget_Train.pkl"))
        accdat_dist_sub = pd.read_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_accdistract_Train.pkl"))
        behdat_sub = pd.read_pickle(bids.direct_results / Path(bids.substring + "motiondiscrim_allbehave_Train.pkl"))

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
            accdat_targ_all = accdat_targ_all.append(accdat_targ_sub, ignore_index=True)  # ignore index just means the index will count all the way up
            accdat_dist_all = accdat_dist_all.append(accdat_dist_sub, ignore_index=True)  # ignore index just means the index will count all the way up
            behdat_all = behdat_all.append(behdat_sub, ignore_index=True)

    # average and plot reaction time data
    behdat_all_avg = behdat_all.groupby(["subID", "Testday", "Attention Type"]).mean()
    behdat_all_avg = behdat_all_avg.reset_index()

    # plot results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    # Reaction time Grouped violinplot
    colors = [settings.red_, settings.orange_, settings.yellow_]

    sns.violinplot(x="Testday", y="Reaction Time", data=behdat_all_avg,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, order = ["Day 1", "Day 2", "Day 3"])
    sns.swarmplot("Testday", y="Reaction Time", data=behdat_all_avg, ax=ax1, color=".2", order = ["Day 1", "Day 2", "Day 3"])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task during NF RT by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    ############## Get Sensitivity
    from scipy.stats import norm
    import math

    Z = norm.ppf  # percentile point function - normal distribution between 0 and 1.

    N_distractors = 896  # number of distractor events per day and condition.
    N_targets = 384  # number of target events per day and condition.

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
    accdat_targ_all.loc[:, "LikelihoodRatio"] = accdat_targ_all.loc[:, "Sensitivity"] * accdat_targ_all.loc[:, "Criterion"]

    # plot results - sensitivity
    plots = ["correct", "miss", "correctreject", "falsealarm"]

    fig, (ax1) = plt.subplots(2, 2, figsize=(10, 10))
    ax1 = ax1.flatten()
    sns.set(style="ticks")
    colors = [settings.red_, settings.orange_, settings.yellow_]
    # Reaction time Grouped violinplot
    for ii in np.arange(4):
        if ii < 2:
            sns.violinplot(x="Testday", y=plots[ii], data= accdat_targ_all,
                           palette=sns.color_palette(colors), ax=ax1[ii])

            sns.swarmplot("Testday", y=plots[ii], data= accdat_targ_all, ax=ax1[ii],color=".2")
        else:
            sns.violinplot(x="Testday", y=plots[ii], data=accdat_dist_all,
                           palette=sns.color_palette(colors), ax=ax1[ii])

            sns.swarmplot("Testday", y=plots[ii], data=accdat_dist_all, ax=ax1[ii], color=".2")
        ax1[ii].spines['top'].set_visible(False)
        ax1[ii].spines['right'].set_visible(False)

        ax1[ii].set_title(plots[ii])

        ax1[ii].set_ylim([0, 100])

    titlestring = 'Motion Task during NF Accuracy by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot Sensitivity results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    colors = [settings.red_, settings.orange_, settings.yellow_]

    sns.violinplot(x="Testday", y="Sensitivity", data=accdat_targ_all,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, order = ["Day 1", "Day 2", "Day 3"])
    sns.swarmplot("Testday", y="Sensitivity", data=accdat_targ_all, ax=ax1, color=".2", order = ["Day 1", "Day 2", "Day 3"])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task during NF Sensitivity by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot Criterion results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    colors = [settings.red_, settings.orange_, settings.yellow_]

    sns.violinplot(x="Testday", y="Criterion", data=accdat_targ_all,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, order = ["Day 1", "Day 2", "Day 3"])
    sns.swarmplot("Testday", y="Criterion", data=accdat_targ_all, ax=ax1, color=".2", order = ["Day 1", "Day 2", "Day 3"])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task during NF Criterion by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # plot LikelihoodRatio results
    fig, (ax1) = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")

    colors = [settings.red_, settings.orange_, settings.yellow_]

    sns.violinplot(x="Testday", y="LikelihoodRatio", data=accdat_targ_all,
                   palette=sns.color_palette(colors), style="ticks", ax=ax1, order = ["Day 1", "Day 2", "Day 3"])
    sns.swarmplot("Testday", y="LikelihoodRatio", data=accdat_targ_all, ax=ax1, color=".2", order = ["Day 1", "Day 2", "Day 3"])

    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    titlestring = 'Motion Task during NF Likelihood Ratio by Day Train ' + settings.string_attntrained[settings.attntrained]
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group / Path(titlestring + '.png'), format='png')

    # Save results out for collation later
    accdat_targ_all['RT'] = behdat_all_avg['Reaction Time']
    accdat_targ_all.to_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[1] + ".pkl"))

if (collate_behaviour_duringNF_compare):

    # get task specific settings
    settings = settings.get_settings_behave_duringNF()

    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):  # cycle trough space and feature train groups
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_behave_duringNF()

        # file names
        bids = helper.BIDS_FileNaming(subject_idx=0, settings=settings, day_val=0)
        print(bids.direct_results_group)

        df_behaveresults_tmp = pd.read_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[1] + ".pkl"))
        df_behaveresults_tmp['AttentionTrained'] = attntrained

        if (attntrainedcount == 0):
            df_behaveresults = df_behaveresults_tmp[['subID','subIDval', 'AttentionTrained', 'Attention Type', 'Testday', 'Sensitivity', 'Criterion', 'correct', 'LikelihoodRatio', 'RT']]
        else:
            df_behaveresults_tmp['subID'] = df_behaveresults_tmp['subID'] + 34
            df_behaveresults = df_behaveresults.append(df_behaveresults_tmp[['subID', 'subIDval', 'AttentionTrained', 'Attention Type', 'Testday', 'Sensitivity', 'Criterion', 'correct', 'LikelihoodRatio', 'RT']])

    # Exclude extremely poor performers.
    tmp = df_behaveresults.groupby('subIDval').mean()['correct']
    exclude = df_behaveresults.groupby('subIDval').mean()[tmp<10] # SUBIDs 1,2,23, 90

    df_behaveresults_clean = df_behaveresults[~df_behaveresults['subID'].isin(exclude['subID'])]

    # # lets run some stats with R - save it out
    df_behaveresults_clean.to_csv(bids.direct_results_group_compare / Path("motiondiscrim_behaveresults_ALL_duringNF.csv"), index=False)


    ##### Plot grouped results across training groups #####
    #### Reaction Time ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal, settings.medteal, settings.darkteal]
    for i in np.arange(2):
        datplot = df_behaveresults_clean[df_behaveresults_clean["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Testday", y="RT", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Testday", y="RT", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task RT training effect by attention during NF'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    #### Sensitivity ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal, settings.medteal, settings.darkteal]
    for i in np.arange(2):
        datplot = df_behaveresults_clean[df_behaveresults_clean["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Testday", y="Sensitivity", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Testday", y="Sensitivity", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task Sensitivity training effect by attention during NF'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    #### Criterion ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal, settings.medteal, settings.darkteal]
    for i in np.arange(2):
        datplot = df_behaveresults_clean[df_behaveresults_clean["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Testday", y="Criterion", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Testday", y="Criterion", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task Criterion training effect by attention during NF'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    #### % Correct ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal, settings.medteal, settings.darkteal]
    for i in np.arange(2):
        datplot = df_behaveresults_clean[df_behaveresults_clean["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Testday", y="correct", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Testday", y="correct", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task Correct training effect by attention during NF'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    #### Likelihood Ratio ####
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Reaction time Grouped violinplot
    colors = [settings.lightteal, settings.medteal, settings.darkteal]
    for i in np.arange(2):
        datplot = df_behaveresults_clean[df_behaveresults_clean["AttentionTrained"] == settings.string_attntrained[i]]

        sns.swarmplot(x="Testday", y="LikelihoodRatio", data=datplot, color="0", alpha=0.3, ax=ax[i])
        sns.violinplot(x="Testday", y="LikelihoodRatio", data=datplot, palette=sns.color_palette(colors), style="ticks", ax=ax[i], inner="box", alpha=0.6)

        ax[i].spines['top'].set_visible(False)
        ax[i].spines['right'].set_visible(False)

        ax[i].set_title(settings.string_attntrained[i])

    titlestring = 'Motion Task LikelihoodRatio training effect by attention during NF'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

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
    sns.violinplot(x="Attention Trained", y= "Selectivity (ΔµV)" , hue = "Testday", data=df_selctivity[df_selctivity["Attention Type"].isin([settings.string_attntrained[0]])], palette=sns.color_palette(colors), ax=ax1, split=True, inner="stick")
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    ax1.set_title(settings.string_attntrained[0] + " Attention")
    ax1.set_ylim(-0.25, 0.65)
    # Accuracy Grouped violinplot
    sns.violinplot(x="Attention Trained", y="Selectivity (ΔµV)", hue="Testday",
                   data=df_selctivity[df_selctivity["Attention Type"].isin([settings.string_attntrained[1]])],
                   palette=sns.color_palette(colors), ax=ax2, split=True, inner="stick")
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
                                            np.isin(settings.subsIDXcollate, np.array([89])))
        settings.num_subs = settings.num_subs - 1
    if (attntrained == 0):  # correct for lost data for sub 21 (feature train)
        settings.subsIDXcollate = np.delete(settings.subsIDXcollate, np.isin(settings.subsIDXcollate, np.array([90])))
        settings.num_subs = settings.num_subs - 1

    # preallocate group mean variables
    num_subs = settings.num_subs +1
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
                                                np.isin(settings.subsIDXcollate, np.array([ 89])))
            settings.num_subs = settings.num_subs - 1
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

    # correct for missing data
    df_rt = df_rt[df_rt["Reaction Time (s)"]< 3]
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
        ax1[attn].set_ylim([0.4, 2.0])
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
    df_rt_SSmean= df_rt_SSmean[df_rt_SSmean["Reaction Time (s)"] < 3]
    sns.violinplot(x="Attention Trained", y="Reaction Time (s)", hue="Testday", data=df_rt_SSmean,
                   palette=sns.color_palette(colors), style="ticks", ax=ax2, split=True, inner="stick")
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    ax2.set_title("Reaction time")
    # ax2.set_ylim([0.4, 2.0])
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
                                                np.isin(settings.subsIDXcollate, np.array([ 89])))
            settings.num_subs = settings.num_subs - 1
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

if (classification_acc_correlations):
    ############# Load Classification Accuracy Data ###############################
    attntrained_vec = []
    sub_vec = []
    classifiertype_vec = []
    classifieracc_vec = []

    # Cycle through trained groups
    for attntrainedcount, attntrained in  enumerate(settings.string_attntrained):
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)

        # get task specific settings
        settings = settings.get_settings_behave_prepost()

        # Cycle through subjects
        for sub_count, sub_val in enumerate(settings.subsIDXcollate):
            # get file names
            bids = helper.BIDS_FileNaming(sub_val, settings, day_val=1)

            # Get Data for Space and Feature Classifier
            for attn_count, attn_val in enumerate(settings.string_attntrained):
                # decide which file to use
                possiblefiles = []
                for filesfound in bids.direct_data_eeg.glob("Classifier_" + attn_val + '_' + bids.filename_eeg + ".mat"):
                    possiblefiles.append(filesfound)
                file2use = possiblefiles[0]

                # load data
                F = h5py.File(file2use, 'r')
                # print(list(F.keys()))

                # get Accuracy
                tmp_acc = np.array(F['ACCURACY_ALL'])*100

                attntrained_vec.append(attntrained)
                sub_vec.append(sub_val)
                classifiertype_vec.append(attn_val)
                classifieracc_vec.append(np.nanmean(tmp_acc))

    # Stack data into a dataframe
    data = {'SubID': sub_vec, 'AttentionTrained': attntrained_vec, 'ClassifierType': classifiertype_vec, 'ClassifierAccuracy': classifieracc_vec}
    df_classifier = pd.DataFrame(data)

    df_classifier_condensed = df_classifier.loc[df_classifier.AttentionTrained==df_classifier.ClassifierType,:].copy()

    # plot results

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.medteal_, settings.lightteal_]

    # Accuracy Grouped violinplot
    sns.violinplot(x="AttentionTrained", y="ClassifierAccuracy", data=df_classifier_condensed  ,
                   palette=sns.color_palette(colors), style="ticks", ax=ax)
    sns.swarmplot(x="AttentionTrained", y="ClassifierAccuracy", data=df_classifier_condensed , color=".5")

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim(0, 100)
    ax.set_title("Accuracy")

    titlestring = 'Classification Accuracy'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    # plot results - both used and unused
    df_classifier_used = df_classifier.loc[df_classifier.AttentionTrained == df_classifier.ClassifierType, :].copy()
    df_classifier_unused = df_classifier.loc[df_classifier.AttentionTrained != df_classifier.ClassifierType, :].copy()

    fig, ax = plt.subplots(1, 1, figsize=(5,5))
    sns.set(style="ticks")
    colors = [settings.medteal_, settings.lightteal_]

    # Accuracy Grouped violinplot
    sns.violinplot(x="ClassifierType", y="ClassifierAccuracy", data=df_classifier,
                   palette=sns.color_palette(colors), style="ticks", ax=ax)
    sns.swarmplot(x="ClassifierType", y="ClassifierAccuracy", data=df_classifier_used, color="0.5")
    sns.swarmplot(x="ClassifierType", y="ClassifierAccuracy", data=df_classifier_unused, color="0.5", order=['Space', 'Feature'])

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.set_ylim(0, 100)
    ax.set_title("Accuracy")

    titlestring = 'Classification Accuracy used vs unused'
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')

    df_classifier[df_classifier['ClassifierType']=='Space'].mean() # 65.744799
    df_classifier[df_classifier['ClassifierType'] == 'Space'].std()  # 12.483115

    df_classifier[df_classifier['ClassifierType']=='Feature'].mean() # 54.375419
    df_classifier[df_classifier['ClassifierType'] == 'Feature'].std()  #  5.715636
    ########## Get SSVEP Selectivity Data #############

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
        settings = settings.get_settings_EEG_prepost()

        # file names
        bids = helper.BIDS_FileNaming(subject_idx=0, settings=settings, day_val=0)

        # load results
        results = np.load(bids.direct_results_group / Path("EEGResults_prepost.npz"), allow_pickle=True)  #

        SSVEPs_epochs_prepost_group = results['SSVEPs_epochs_prepost_group']  # results['SSVEPs_prepost_group']
        diffdat = SSVEPs_epochs_prepost_group[0, :, :, :] - SSVEPs_epochs_prepost_group[1, :, :, :]  # [day,attn,sub]

        # store results for attention condition
        tmp = [settings.string_prepost[0]] * settings.num_subs * settings.num_attnstates + [
            settings.string_prepost[1]] * settings.num_subs * settings.num_attnstates
        daystrings = np.concatenate((daystrings, tmp))

        tmp = [settings.string_attntrained[0]] * settings.num_subs + [
            settings.string_attntrained[1]] * settings.num_subs
        attntaskstrings = np.concatenate((attntaskstrings, tmp, tmp))

        tmp = [settings.string_attntrained[
                   attntrained]] * settings.num_subs * settings.num_days * settings.num_attnstates
        attnstrings = np.concatenate((attnstrings, tmp))

        tmp = np.concatenate((diffdat[0, 0, :], diffdat[0, 1, :], diffdat[1, 0, :], diffdat[1, 1, :]))
        selectivity_compare = np.concatenate((selectivity_compare, tmp))

    data = {'Testday': daystrings, 'Attention Type': attntaskstrings, 'Attention Trained': attnstrings,
            'Selectivity (ΔµV)': selectivity_compare}
    df_selctivity = pd.DataFrame(data)

    # Add selectivity - get the indices we're interested in.
    idx_space_pre = np.logical_and(df_selctivity.Testday == 'pre-training',df_selctivity["Attention Type"] == "Space")
    idx_feature_pre = np.logical_and(df_selctivity.Testday == 'pre-training', df_selctivity["Attention Type"] == "Feature")

    idx_space_post = np.logical_and(df_selctivity.Testday == 'post-training', df_selctivity["Attention Type"] == "Space")
    idx_feature_post = np.logical_and(df_selctivity.Testday == 'post-training', df_selctivity["Attention Type"] == "Feature")

    # create new correlation dataframe to add selectivity to
    df_correlationsdat = df_classifier_condensed.drop(["ClassifierType"], axis = 1).copy().reset_index()
    df_correlationsdat = df_correlationsdat.drop(["index"], axis=1)

    # add selectivity - pre and post training
    df_correlationsdat["Space_Selectivity_pre"] = df_selctivity.loc[idx_space_pre,:].reset_index().loc[:,"Selectivity (ΔµV)"]
    df_correlationsdat["Feature_Selectivity_pre"] = df_selctivity.loc[idx_feature_pre,:].reset_index().loc[:,"Selectivity (ΔµV)"]

    df_correlationsdat["Space_Selectivity_post"] = df_selctivity.loc[idx_space_post, :].reset_index().loc[:, "Selectivity (ΔµV)"]
    df_correlationsdat["Feature_Selectivity_post"] = df_selctivity.loc[idx_feature_post, :].reset_index().loc[:, "Selectivity (ΔµV)"]

    # Add training effect
    df_correlationsdat["Space_Selectivity_trainefc"] =  df_selctivity.loc[idx_space_post, :].reset_index().loc[:, "Selectivity (ΔµV)"] - df_selctivity.loc[idx_space_pre, :].reset_index().loc[:, "Selectivity (ΔµV)"]
    df_correlationsdat["Feature_Selectivity_trainefc"] = df_selctivity.loc[idx_feature_post, :].reset_index().loc[:,"Selectivity (ΔµV)"] - df_selctivity.loc[idx_feature_pre, :].reset_index().loc[:,"Selectivity (ΔµV)"]


    ######## Load behavioural data
    # cycle trough space and feature train groups
    for attntrainedcount, attntrained in enumerate(settings.string_attntrained):  # cycle trough space and feature train groups
        # setup generic settings
        settings = helper.SetupMetaData(attntrainedcount)
        settings = settings.get_settings_behave_prepost()

        # file names
        bids = helper.BIDS_FileNaming(subject_idx=0, settings=settings, day_val=0)

        # accdat_all_avg.to_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))
        df_behaveresults_tmp = pd.read_pickle(bids.direct_results_group / Path("motiondiscrim_behaveresults_" + settings.string_testtrain[0] + ".pkl"))
        df_behaveresults_tmp['AttentionTrained'] = attntrained

        if (attntrainedcount==0):
            df_behaveresults = df_behaveresults_tmp[['AttentionTrained', 'Testday', 'Attention Type', 'Sensitivity', 'Criterion', 'RT']]
        else:
            df_behaveresults = df_behaveresults.append(df_behaveresults_tmp[['AttentionTrained', 'Testday', 'Attention Type', 'Sensitivity', 'Criterion', 'RT']])

    # Add data to correlations dat - get indices
    idx_space_pre = np.logical_and(df_behaveresults.Testday == 'Day 1', df_behaveresults["Attention Type"] == "Space")
    idx_feature_pre = np.logical_and(df_behaveresults.Testday == 'Day 1', df_behaveresults["Attention Type"] == "Feature")

    idx_space_post = np.logical_and(df_behaveresults.Testday == 'Day 4', df_behaveresults["Attention Type"] == "Space")
    idx_feature_post = np.logical_and(df_behaveresults.Testday == 'Day 4', df_behaveresults["Attention Type"] == "Feature")

   # Add data to correlations dat
    dattype = "Sensitivity"
    df_correlationsdat["Space_" + dattype + "_pre"] = df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_pre"] = df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:, dattype]

    df_correlationsdat["Space_" + dattype + "_post"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_post"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:, dattype]

    # Add training effect
    df_correlationsdat["Space_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:, dattype] - df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:, dattype] - df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:, dattype]

    # Add data to correlations dat
    dattype = "Criterion"
    df_correlationsdat["Space_" + dattype + "_pre"] = df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_pre"] = df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:, dattype]

    df_correlationsdat["Space_" + dattype + "_post"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_post"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:, dattype]

    # Add training effect
    df_correlationsdat["Space_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:, dattype] - df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:, dattype] - df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:, dattype]

    # Add data to correlations dat
    dattype = "RT"
    df_correlationsdat["Space_" + dattype + "_pre"] = df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_pre"] = df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:, dattype]

    df_correlationsdat["Space_" + dattype + "_post"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_post"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:, dattype]

    # Add training effect
    df_correlationsdat["Space_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_space_post, :].reset_index().loc[:, dattype] - df_behaveresults.loc[idx_space_pre, :].reset_index().loc[:, dattype]
    df_correlationsdat["Feature_" + dattype + "_trainefc"] = df_behaveresults.loc[idx_feature_post, :].reset_index().loc[:, dattype] - df_behaveresults.loc[idx_feature_pre, :].reset_index().loc[:, dattype]

    # plot classification accuracy vs. Selectivity
    import scipy.stats as stats

    i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_Selectivity_pre"]]
    k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_Selectivity_pre"]]
    corrs_space = stats.pearsonr(i["ClassifierAccuracy"], j["Space_Selectivity_pre"])
    corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_Selectivity_pre"])
    corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"], k["ClassifierAccuracy"]], axis=0), pd.concat([j["Space_Selectivity_pre"], l["Feature_Selectivity_pre"]], axis=0))

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="ClassifierAccuracy", y="Space_Selectivity_pre", ax=ax, color=settings.yellow_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy", y="Feature_Selectivity_pre", ax=ax, color=settings.lightteal_)
    ax.set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    ax.set_ylabel("Selectivity for trained attention type")
    ax.legend(settings.string_attntrained, title="Attention Trained")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    titlestring = "Classifier Acc Vs. Selectivity Scatter"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    # Plot how training effects relate to classifcation accuracy

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    # Selectivity
    i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_Selectivity_trainefc"]]
    k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_Selectivity_trainefc"]]
    corrs_space = stats.pearsonr(i["ClassifierAccuracy"], j["Space_Selectivity_trainefc"])
    corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_Selectivity_trainefc"])
    corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"],k["ClassifierAccuracy"]], axis=0), pd.concat([j["Space_Selectivity_trainefc"],l["Feature_Selectivity_trainefc"]], axis=0))

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy", y="Feature_Selectivity_trainefc", ax=ax[0], color = settings.lightteal_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained=="Space"], x="ClassifierAccuracy", y="Space_Selectivity_trainefc",  ax=ax[0], color=settings.yellow_)
    ax[0].set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    # ax[0].set_title("Classifier Acc Vs. Selectivity train effect")
    ax[0].set_ylabel("change in Selectivity for trained attention type")
    ax[0].legend(['Feature', 'Space'], title="Attention Trained")
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    # Sensitivity
    i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_Sensitivity_trainefc"]]
    k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_Sensitivity_trainefc"]]
    corrs_space =stats.pearsonr(i["ClassifierAccuracy"],j["Space_Sensitivity_trainefc"])
    corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_Sensitivity_trainefc"])
    corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"], k["ClassifierAccuracy"]], axis=0), pd.concat([j["Space_Sensitivity_trainefc"], l["Feature_Sensitivity_trainefc"]], axis=0))

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy", y="Feature_Sensitivity_trainefc", ax=ax[1], color=settings.lightteal_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="ClassifierAccuracy", y="Space_Sensitivity_trainefc", ax=ax[1], color=settings.yellow_)
    ax[1].set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    # ax[1].set_title("Classifier Acc Vs. Sensitivity (d') train effect")
    ax[1].set_ylabel("Change in Sensitivity for trained attention type")
    ax[1].legend(['Feature', 'Space'], title="Attention Trained")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)
    #
    # # Criterion
    # i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    # j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_Criterion_trainefc"]]
    # k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    # l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_Criterion_trainefc"]]
    # corrs_space = stats.pearsonr(i["ClassifierAccuracy"], j["Space_Criterion_trainefc"])
    # corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_Criterion_trainefc"])
    # corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"], k["ClassifierAccuracy"]], axis=0), pd.concat([j["Space_Criterion_trainefc"], l["Feature_Criterion_trainefc"]], axis=0))
    #
    # sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy", y="Feature_Criterion_trainefc", ax=ax[1][0], color=settings.lightteal_)
    # sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="ClassifierAccuracy", y="Space_Criterion_trainefc", ax=ax[1][0], color=settings.yellow_)
    # ax[1][0].set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    # # ax[1].set_title("Classifier Acc Vs. Sensitivity (d') train effect")
    # ax[1][0].set_ylabel("Change in Criterion for trained attention type")
    # ax[1][0].legend(['Feature', 'Space'], title="Attention Trained")
    # ax[1][0].spines['top'].set_visible(False)
    # ax[1][0].spines['right'].set_visible(False)
    #
    # # Criterion
    # i = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["ClassifierAccuracy"]]
    # j = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Space", ["Space_RT_trainefc"]]
    # k = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["ClassifierAccuracy"]]
    # l = df_correlationsdat.loc[df_correlationsdat.AttentionTrained == "Feature", ["Feature_RT_trainefc"]]
    # corrs_space = stats.pearsonr(i["ClassifierAccuracy"], j["Space_RT_trainefc"])
    # corrs_feat = stats.pearsonr(k["ClassifierAccuracy"], l["Feature_RT_trainefc"])
    # corrs_both = stats.pearsonr(pd.concat([i["ClassifierAccuracy"], k["ClassifierAccuracy"]], axis=0), pd.concat([j["Space_RT_trainefc"], l["Feature_RT_trainefc"]], axis=0))
    #
    # sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="ClassifierAccuracy", y="Feature_RT_trainefc", ax=ax[1][1], color=settings.lightteal_)
    # sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="ClassifierAccuracy", y="Space_RT_trainefc", ax=ax[1][1], color=settings.yellow_)
    # ax[1][1].set_title("r = " + str(corrs_both[0]) + ', p = ' + str(corrs_both[1]))
    # # ax[1].set_title("Classifier Acc Vs. Sensitivity (d') train effect")
    # ax[1][1].set_ylabel("Change in RT for trained attention type")
    # ax[1][1].legend(['Feature', 'Space'], title="Attention Trained")
    # ax[1][1].spines['top'].set_visible(False)
    # ax[1][1].spines['right'].set_visible(False)
    #

    titlestring = "Effect of classifier accuracy on training"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    # plot Retest Reliability of measures
    fig, ax = plt.subplots(2, 2, figsize=(12, 12))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=df_correlationsdat, x="Space_Selectivity_pre", y="Space_Selectivity_post", hue="AttentionTrained",palette=sns.color_palette(colors), ax=ax[0][0])
    ax[0][0].set_title("Space Selectivity Pre Vs. Post")

    sns.scatterplot(data=df_correlationsdat, x="Feature_Selectivity_pre", y="Feature_Selectivity_post", hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[0][1])
    ax[0][1].set_title("Feature Selectivity Pre Vs. Post")

    sns.scatterplot(data=df_correlationsdat, x="Space_Sensitivity_pre", y="Space_Sensitivity_post", hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[1][0])
    ax[1][0].set_title("Space Sensitivity Pre Vs. Post")

    sns.scatterplot(data=df_correlationsdat, x="Feature_Sensitivity_pre", y="Feature_Sensitivity_post", hue="AttentionTrained", palette=sns.color_palette(colors), ax=ax[1][1])
    ax[1][1].set_title("Feature Selectivity Pre Vs. Post")

    titlestring = "Retest reliability of attention measures"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')


    # Plot Construct validity

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    sns.set(style="ticks")
    colors = [settings.yellow_, settings.lightteal_]

    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Space"], x="Space_Selectivity_pre", y="Space_Sensitivity_pre", ax=ax[0], color=settings.yellow_)
    sns.scatterplot(data=df_correlationsdat[df_correlationsdat.AttentionTrained == "Feature"], x="Feature_Selectivity_pre", y="Feature_Sensitivity_pre", ax=ax[0], color = settings.lightteal_)
    ax[0].set_title("SSVEP Selectivity Vs. Behave Sensitivity (pre train)")
    ax[0].set_ylabel("Sensitivity (d')")
    ax[0].set_xlabel("SSVEP Selectivity")
    ax[0].legend(settings.string_attntrained, title="Attention Trained")
    ax[0].spines['top'].set_visible(False)
    ax[0].spines['right'].set_visible(False)

    sns.scatterplot(data=df_correlationsdat, x="Space_Selectivity_pre", y="Feature_Selectivity_pre", hue="AttentionTrained",palette=sns.color_palette(colors), ax=ax[1])
    ax[1].set_title("Selectivity Space Vs Feature (pre train)")
    ax[1].set_ylabel("Feature_Selectivity_pre")
    ax[1].set_xlabel("Space_Selectivity_pre")
    ax[1].spines['top'].set_visible(False)
    ax[1].spines['right'].set_visible(False)

    sns.scatterplot(data=df_correlationsdat, x="Space_Sensitivity_pre", y="Feature_Sensitivity_pre", hue="AttentionTrained",palette=sns.color_palette(colors), ax=ax[2])
    ax[2].set_title("Sensitivity Space Vs Feature (pre train)")
    ax[2].set_ylabel("Feature_Sensitivity_pre")
    ax[2].set_xlabel("Space_Sensitivity_pre")
    ax[2].spines['top'].set_visible(False)
    ax[2].spines['right'].set_visible(False)

    titlestring = "Construct validity"
    plt.suptitle(titlestring)
    plt.savefig(bids.direct_results_group_compare / Path(titlestring + '.png'), format='png')
