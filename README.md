# Implicit decoded neurofeedback enhances spatial and feature-based visual selective attention by reducing excitatory/inhibitory balance

This methods for this project were pre-registered and can be found at https://osf.io/q4whu/.

MATLAB code used to control the experiment,  Python code used to analyse the data, and an R Markdown document detailing the statistical analyses can be found at https://github.com/air2310/AttentionNeurofeedbackAnalysis. 

The abstract for the planned manuscript reads as follows:<br>
*"Mechanisms of attention are crucial for regulating the flow of visual information in the service of adaptive behaviour. Here we asked whether the efficacy of human attention can be altered using real-time, decoded neurofeedback targeting spatial and feature-based selection networks. Participants underwent a four-day training procedure, receiving task-based neurofeedback yoked in real-time to their spatial attention, feature-based attention, or sham feedback yoked to the attentional state of a different participant. Participants were blind to the training goals. Compared with sham, active neurofeedback elicited greater shifts in neural task representations and larger behavioural performance gains on a separate attention test task. Enhanced behavioural performance was mediated by a shift in excitatory/inhibitory (E/I) balance toward greater inhibition for both active training groups. In turn, behavioural training benefits transferred to the untrained mode of attention. These results implicate E/I balance as a common neural mechanism of spatial and feature-based attention and highlight the importance of considering generalised training effects arising from common mechanisms as an important consideration for future research employing decoded neurofeedback."*


## Details related to access to the data

### Data user agreement

These data and this code is published with a CC BY 4.0 License

### Contact person

Angela Renton <br>
angie.renton23@gmail.com <br>
https://orcid.org/0000-0003-4815-9056

## Overview

### Project name
Real-time decoded neurofeedback enhances spatial and feature-based visual selective attention by reducing excitatory/inhibitory balance

### Year(s) that the project ran
2020-2021 <br>
Data were collected between 9am and 5pm. Each participant performed all test days at the same time of day. 

### Brief overview of the tasks in the experiment

In this project we sought to assess the cognitive specificity of an implicit decoded neurofeedback protocol for training spatial and feature-based visual selective attention. If participants can learn to regulate only the cognitive function most strongly associated with decoder-based feedback, we expect that feedback designed to enhance one mode of attention (e.g., spatial) should not influence the other (feature). By contrast, if feedback is also integrated more broadly toward other cognitive functions, then neurofeedback aiming to enhance one mode of attention should also improve the other. To assess these possibilities, we developed a four-day EEG-based training protocol to enhance the patterns of neural activity associated with visual selective attention in three separate groups of participants (spatial neurofeedback group, feature-based neurofeedback group, sham control group, Total N = 108). 

Neurofeedback was delivered through dynamic changes in the difficulty of an attentional cueing task. Participants were tasked with identifying bursts of coherent motion in a frequency-tagged display containing fields of black and white moving dots while we recorded their brain activity using EEG. Attentional cues on each trial indicated that participants should monitor a subset of the dots, defined by the intersection of a cued colour and spatial location, for brief bursts of coherent motion. To generate neurofeedback, frequency-tagged EEG data were passed in real-time to machine learning classifiers trained to classify either the attended colour (feature neurofeedback group), or spatial configuration (space neurofeedback group, Figure 1b). Neurofeedback representing the output of the classifier was presented by modulating the contrast of phase-scrambled noise masks presented behind the dot fields, where increased contrast made target bursts of motion direction more difficult to discriminate . Critically the contrast of the noise masks varied over time but was always matched across locations. Thus, neurofeedback training reinforced endogenous neural states associated with strong attentional selectivity, rather than exogenously eliciting those states through changes in salience across targets (i.e. top-down rather than bottom-up attention). 

To assess the efficacy and specificity of this neurofeedback training, participants performed an attention test before and after neurofeedback training. This task was similar in form to the training task but did not contain any phase-scrambled noise masks. Further, while we used combined spatial/feature-based cues during neurofeedback, participants were cued to independently engage either spatial or feature-based attention during the attention test. A third “sham neurofeedback” group of participants each completed the same protocol but were shown the feedback which had previously been generated for a different active neurofeedback participant. As such, all three neurofeedback groups viewed identical displays and received identical instructions. Participants were blind to the existence of separate neurofeedback groups and to the intended purpose of neurofeedback training. Thus, any differences between groups could only be attributed to the nature of the link between task-difficulty and attentional state.


### Description of the contents of the dataset

The dataset contains data from three groups of subjects (Feature-based attnetion training, spatial attention training, sham training). Each group's data are in their own folder ('TrainFeature', 'TrainSpace', 'TrainSham'). Each folder contains BIDS formatted data for each subject in the experiment. These folders contain EEG and behavioural data for each of the four experimental test days. 

See the published manuscript for a full description of the variables in this dataset. 

## Methods

### Subjects

*Participant metadata.* 
N = 132 individuals (97 females, age M = 23.30 years, SD = 4.77) volunteered to participate in the experiment after providing informed consent. Of these participants, 37 received feature-based attention neurofeedback (26 females, age M = 23.73 years, SD = 5.45) and 34 participants received spatial attention neurofeedback (27 females, age M = 23.57 years, SD = 3.94). N = 37 participants received sham neurofeedback training (29 females, age M = 23.08 years, SD = 5.77). N = 24 participants could not be classified in either the feature-based or spatial attention conditions and therefore did not proceed to the training phase of the study (15 females, age M = 22.75 years, SD = 2.85). This attrition rate can likely be attributed to both individual differences in cortical folding and common sources of noise in EEG, which may make some participants more difficult to classify than others.

*Recruitment details.* 
Participants were recruited through The University of Queensland’s School of Psychology paid research participation scheme. All participants were neurotypical, had normal or corrected to normal vision, and were screened for a personal and/or family history of photosensitive epilepsy, which is contraindicated for flickering visual displays. Participants were further excluded from neurofeedback training if their attended feature or spatial position could not be classified significantly greater than chance (Student’s t-test). Participants were paid $10 per half hour (i.e., $100 in total for neurofeedback training over 4 consecutive days, or $20 in total for those who could not be classified and therefore did not proceed to the training phase). The study was approved by The University of Queensland Human Research Ethics Committee, and the experiment was performed according to the relevant guidelines and regulations.

*Stopping rule.* 
An a priori declaration was made that once a minimum of 30 participants had been tested per condition, data collection would continue until the Bayes factor for the attentional selectivity training effect was greater than three in favour of either the null or the alternative hypothesis for each neurofeedback group, up to a limit of 40 participants per group62,83. See the Offline EEG analysis: SSVEP Selectivity section below for details on the calculation of this training effect. 

### Apparatus

*Display computer specifications.*
All displays were presented at a viewing distance of 57 cm on a 24-inch ASUS VG248QE monitor with a refresh rate of 144 Hz and resolution of 1920 × 1080. Stimuli were presented using the Psychophysics Toolbox 87 running in MATLAB R2017a (64-bit) under Windows 10 (64-bit). The experiment was run on a Dell Precision Tower 5810 desktop computer containing an Intel Xeon E7-4809 v2 CPU and NVIDIA QUADRO M4000 GPU.

*EEG recording.*
EEG data were sampled at 1200 Hz using a g.USBamp amplifier (g.tec Medical Engineering, GmbH, Austria) from 9 active Ag/AgCl scalp electrodes arranged according to the international standard 10-20 system for electrode placement in a nylon head cap88. The electrode positions were as follows: Iz, Oz, O1, O2, POz, PO3, PO4 PO7, PO8. This group of electrodes was chosen because SSVEPs are most strongly represented at occipitoparietal electrode sites 65. The ground electrode was positioned at FCz, and an active Ag/AgCl ear clip electrode was used as the reference. EEG data were filtered in real time with a notch filter at 48-52 Hz and a 1-100 Hz bandpass filter. The experiment was run across three instances of MATLAB. EEG data were originally read into the first instance, where data were stored for saving. During neurofeedback, these data were sent to a second instance of MATLAB for real-time processing and classification. The resulting stream of data, representing classification of the attended feature and spatial location, were then sent to the third instance of MATLAB from which the experimental task was displayed. EEG data were streamed between MATLAB instances using the FieldTrip buffer. 


### Task Protocol

The training protocol ran over four consecutive days and consisted of three days of neurofeedback training and one post-training assessment day. Sessions were timed such that each participant’s testing and training occurred at approximately the same time each day. 

*Day 1.* When participants arrived at the lab for the first time, they provided informed consent and were fitted with an EEG cap. All participants began the study by performing the pre-training attention test, which is described in detail below. This test was used to assess neural and behavioural measures of feature-based and spatial attention, as well as to obtain data on which to train the classifiers. 

Pilot testing prior to the main study showed that it was more difficult for participants to discriminate between cued and un-cued features than between cued and un-cued spatial locations using a classifier trained on single trial SSVEPs. Participants were therefore allocated to active neurofeedback groups as follows: If participants’ feature-based attention could be classified significantly better than chance (based on a single-sample Student's t-test), they were allocated to the feature neurofeedback group. If not, a classifier was trained on participants’ spatial attention. If participants’ spatial attention could be classified significantly better than chance, they were allocated to the space neurofeedback group. If neither feature-based nor spatial attention could be classified, participants were excluded from the remainder of the study. Classifiers were not trained on control group (sham) participants, as these individuals did not receive real-time neurofeedback. The three groups experienced identical protocols, with the only difference being whether they underwent feature-based, spatial or sham neurofeedback. After the pre-training attention test, participants performed two transfer tasks: a simple visual search task and a visual-verbal n-back task. 

After these pre-training measures, participants proceeded to their first neurofeedback training session. Neurofeedback was provided as part of an interactive attentional cueing task, which was a modified version of the test task used to collect classifier training data (described in detail below). Participants were aware that feedback represented how well they were processing the display, but were not aware of the specifics of the feedback calculation, or of which neurofeedback group they were in. 

*Days 2-3.* For the following two days, participants were fitted with EEG caps and performed the motion-discrimination neurofeedback training task. 

*Day 4.* On the final day of the study, participants were fitted with EEG caps and retested on the Day 1 (pre-training) tasks. Participants began by performing the post-training attention test, followed by the visual search task and the visual-verbal n-back task. 


### Task details

The experiment aimed to specifically train participants to enhance patterns of neural activity associated with either spatial or feature-based attention. As such, participants were split into three neurofeedback groups: (1) a spatial attention training group, (2) a feature-based attention training group, and (3) a control group who received sham neurofeedback. These participants each underwent a three-day neurofeedback training protocol in which the difficulty of an ongoing attention task was continuously modulated according to participants’ real-time decoded attentional state (active neurofeedback), or the decoded attentional state of a different past participant (sham). In this task, participants viewed four separate circular patches positioned in each of the four quadrants, and located equidistant from a central fixation cross. Each patch contained both black and white randomly moving dots. Periodically, one of the eight sets of moving dots (e.g., black, upper-left quadrant) moved coherently in one of four different motion directions (0°, 90°,180°, 270°). On each trial, a cue at fixation indicated that participants should monitor for these coherent motion epochs in a subset of the randomly moving dots, defined by the intersection of colour (black or white) and spatial position. To deter participants from shifting their gaze to a specific quadrant, the central cue always indicated two diagonally opposite spatial positions. Thus, to successfully perform the task, participants needed fixate at the centre of the display and covertly hold their attention to the cued feature at diagonally opposing spatial locations. Participants were asked to respond as quickly and accurately as possible with a button press indicating the motion direction of each target (a burst of coherent motion in the cued dots), while ignoring distractors (a burst of coherent motion in the un-cued dots). Dots of each colour (black, white) for each spatial configuration (upper-left and lower-right or upper-right and lower-left) flickered at different frequencies (4.5 Hz, 8.0 Hz, 12 Hz, 14.4 Hz) to elicit SSVEPs, allowing for real-time tracking of participants’ attentional selectivity for these visual stimuli. 

While participants performed the attention task, their SSVEPs were parsed by a machine learning classifier to determine whether they were consistently attending to the dots in the cued spatial configuration (spatial training), or to the dots carrying the cued feature (feature-based training; Figure 1a). Based on this classification, the contrast of phase-scrambled noise masks presented behind the dot patches was dynamically adjusted, with the consequence that the task became easier when participants achieved and held the prescribed attentional state (Figure 1b). If participants maintained this state for longer than 300 ms, the contrast shifted further to make the task even easier (lower contrast noise) or more difficult (higher contrast noise). Notably, these changes affected all coloured dots and locations equally, so that the physical displays in each of the four quadrants were always of equal salience. Thus, neurofeedback did not encourage exogenous (bottom-up) shifts in attention by altering the relative reliability or salience of the visual stimuli themselves. Rather, neurofeedback rewarded patterns of neural activity associated with strong spatial or feature-based attentional selectivity through changes in overall task-difficulty. 

While the central cue indicated that participants should attend to both a specified colour and pair of spatial locations, feedback for the feature neurofeedback group was generated using a classifier trained to discriminate between attention to one feature or the other, across all spatial positions. Likewise, feedback for the spatial neurofeedback group was generated using a classifier trained to discriminate between attention to one pair of spatial locations over the other, across all visual features at those positions. Thus, neurofeedback specifically encouraged one attentional strategy over the other without modulating the relative salience of any spatial location or feature; training was achieved through reinforcement of endogenous states rather than by exogenously evoking the target state.  Participants in the sham control condition were each matched to an active neurofeedback participant (equal numbers from space and feature-neurofeedback groups) and were shown the feedback intended for that participant. Thus, the only difference in the experimental protocol between the three neurofeedback groups was the contingency regulating the contrast changes. Further, while all participants (including the sham group) were informed that the contrast changes were linked to their recorded neural activity in real-time, they were unaware of the nature of this relationship and were blind to the existence of separate neurofeedback groups. 

To assess how neurofeedback training impacted visual attention, participants were tested before and after training on a separate, feedback-free version of the attentional task, as well as on a visual search task and a visual-verbal n-back task, to test for any transfer effects. 


### Experimental location

These data were collected in the Brain Computer Interface EEG laboratory located at <br>
The University of Queensland <br>
School of Psychology <br>
St Lucia, QLD, 4072 <br>
Australia. 

### Missing data

Due to technical issues, some of the earlier participants are missing EEG files for one of the Neurofeedback days. 

## Code usage
The analyses for this project are orchestrated through the python file "main_ATTNNF.py". Each specific analysis is implemented as a function which is imported within this script. 
