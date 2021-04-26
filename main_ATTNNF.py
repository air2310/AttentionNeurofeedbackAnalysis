# Import necessary packages
import pandas as pd
import helperfunctions_ATTNNF as helper
import functions_getEEGprepost as geegpp
import functions_getEEG_duringNF as geegdnf
import analyse_visualsearchtask as avissearch
import analyse_nbacktask as anback
import analyse_motiontask as analyse_motiontask
import analyseNeurofeedback as analyse_NF
import CorrelationAnalyses as analyse_corr

#### New Analyses
# TODO: exclude chance level vis search Ps
# TODO: look at differences between classifiable and unclassifiable participants.
# TODO: analyse wavelets around movement epochs
# TODO: individual trial independance of spatial and feature based-attention measure - how correlated are they really?
# TODO: Sync to main thread
# TODO: fix interping
# look at ERP
# What about just the people who did really well?
# fix up channel interp.

# setup generic settings
attntrained = 2  # 0 = Space, 1 = Feature, 2 = Sham
settings = helper.SetupMetaData(attntrained)
print("Analysing Data for condition train: " + settings.string_attntrained[settings.attntrained])


######## Decide which single subject analyses to do ########

# analyse_behaviour_prepost = True # Analyse Behaviour Pre Vs. Post Training
analyse_behaviour_prepost = False  # Analyse Behaviour Pre Vs. Post Training

# analyse_behaviour_duringNF = True # Analyse Behaviour during Training
analyse_behaviour_duringNF = False  # Analyse Behaviour duringTraining

# analyse_EEG_prepost =True # analyse EEG Pre Vs. Post Training
analyse_EEG_prepost = False  # analyse EEG Pre Vs. Post Training

# analyse_EEG_duringNF = True # analyse EEG during Neurofeedback
analyse_EEG_duringNF = False  # analyse EEG during Neurofeedback

# analyse_visualsearchtask = True # Analyse Visual Search Task
analyse_visualsearchtask = False  # Analyse Visual Search Task
#
# analyse_nbacktask = True # Analyse N-back Task
analyse_nbacktask = False  # Analyse N-back Task
#
# analyse_Neurofeedback = True # Analyse Neurofeedback and sustained attention
analyse_Neurofeedback = False  # Analyse Neurofeedback and sustained attention


######## Decide which group analyses to do ########

# collate_behaviour_prepost = True  # Collate Behaviour Pre Vs. Post Training
collate_behaviour_prepost = False  # Collate Behaviour Pre Vs. Post Training
#
# collate_behaviour_duringNF = True # Collate Behaviour during Training
collate_behaviour_duringNF = False  # Collate Behaviour during Training

# collateEEGprepost = True # Collate EEG Pre Vs. Post Training across subjects
collateEEGprepost = False  # Collate EEG Pre Vs. Post Training across subjects

# collateEEG_duringNF = True  # Collate EEG during Neurofeedback
collateEEG_duringNF = False  # Collate EEG during Neurofeedback
#
# collate_visualsearchtask = True # Collate Visual Search results
collate_visualsearchtask = False  # Collate Visual Search results

# collate_nbacktask = True  # Analyse N-back Task
collate_nbacktask = False  # Analyse N-back Task


######## Decide which group comparison analyses to do ########

# classification_acc_correlations = True # Assess whether classification accuracy correlated with training effects
classification_acc_correlations = False  # Assess whether classification accuracy correlated with training effects

# collate_Neurofeedback = True # collate Neurofeedback and sustained attention
collate_Neurofeedback = False  # collate Neurofeedback and sustained attention

collate_behaviour_prepost_compare = True # Collate Behaviour Pre Vs. Post Training compare training groups
# collate_behaviour_prepost_compare = False  # Collate Behaviour Pre Vs. Post Training compare training groups

# collate_behaviour_duringNF_compare = True # Collate Behaviour during Training compare training groups
collate_behaviour_duringNF_compare = False  # Collate Behaviour during Training compare training groups

# collateEEGprepostcompare = True # Collate EEG Pre Vs. Post Training across subjects
collateEEGprepostcompare = False  # Collate EEG Pre Vs. Post Training across subjects


# Some settings for how things will run
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', 10)


##### iterate through subjects for individual subject analyses #####

for sub_count, sub_val in enumerate(settings.subsIDX):
    print(sub_val)

    if analyse_behaviour_prepost:
        test_train = 0
        analyse_motiontask.run(settings, sub_val, test_train)

    if analyse_behaviour_duringNF:
        test_train = 1
        analyse_motiontask.run(settings, sub_val, test_train)

    if analyse_Neurofeedback:
        analyse_NF.run(settings, sub_val)

    if analyse_EEG_prepost:
        geegpp.analyseEEGprepost(settings, sub_val)

    if analyse_EEG_duringNF:
        geegdnf.analyseEEG_duringNF(settings, sub_val)

    if analyse_visualsearchtask:
        avissearch.analyse_visualsearchtask(settings, sub_val)

    if analyse_nbacktask:
        anback.analyse_nbacktask(settings, sub_val)


##### Collate results across all subjects analyses #####

# Collate pre-post motion task behaviour
if collate_behaviour_prepost:
    analyse_motiontask.collate_behaviour_prepost(settings)

# Compare pre-post motion task behaviour between conditions.
if collate_behaviour_prepost_compare:
    analyse_motiontask.collate_behaviour_prepost_compare(settings)

# Collate during Neurofeedback motion task behaviour
if collate_behaviour_duringNF:
    analyse_motiontask.collate_behaviour_duringNF(settings)

# Compare during Neurofeedback motion task behaviour between conditions.
if collate_behaviour_duringNF_compare:
    analyse_motiontask.collate_behaviour_duringNF_compare(settings)

# Collate the neurofeedback we presented.
if collate_Neurofeedback:
    analyse_NF.collate_Neurofeedback(settings)

# Collate pre-post motion task EEG
if collateEEGprepost:
    geegpp.collateEEGprepost(settings)

# Compare pre-post motion task EEG between training conditions
if collateEEGprepostcompare:
    geegpp.collateEEGprepostcompare(settings)

# Collate during Neurofeedback motion task EEG
if collateEEG_duringNF:
    geegdnf.collateEEG_duringNF(settings)

# Collate Visual Search Task Behaviour
if collate_visualsearchtask:
    avissearch.collate_visualsearchtask(settings)

# Collate N-Back Task Behaviour
if collate_nbacktask:
    anback.collate_nbacktask(settings)

if classification_acc_correlations:
    analyse_corr.classification_acc_correlations(settings)