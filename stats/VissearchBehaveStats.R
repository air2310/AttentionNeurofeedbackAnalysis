library(dplyr)
library(BayesFactor)
library(car)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
behavedata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/VISSEARCH_behaveresults_ALL.csv")

#### Step 2 - create factors

behavedata_orig = behavedata

behavedata$Attention.Trained = factor(behavedata$Attention.Trained)
levels(behavedata$Attention.Trained) = c( "Feature", "Sham","Space")

behavedata$NFTraining = behavedata$Attention.Trained
levels(behavedata$NFTraining) = c( "Neurofeedback", "Sham","Neurofeedback")


behavedata$Testday = factor(behavedata$Testday)
levels(behavedata$Testday) = c("post-training", "pre-training")


behavedata$Set.Size = factor(behavedata$Set.Size)
levels(behavedata$Set.Size) = c("SS12", "SS16", "SS8" )


behavedata$SubID = factor(behavedata$SubID)

# Calculate training effects


idx_d1 =behavedata$Testday=="pre-training"
idx_d4 =behavedata$Testday=="post-training"

tmp_d1 = behavedata[idx_d1,]
tmp_d4 = behavedata[idx_d4,]

behavedata_train = tmp_d1
behavedata_train$AccuracyTrEfct = tmp_d4$Accuracy.... - tmp_d1$Accuracy....
behavedata_train$RT_TrEfct = tmp_d4$Reaction.Time..s. - tmp_d1$Reaction.Time..s.


## ACCURACY
# step 1 - check if set size interacts with training effects
## Test effects of NF, with set size considered

bf = anovaBF(AccuracyTrEfct ~ Attention.Trained * Set.Size + SubID , data=behavedata_train, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?

# display results
bf
bfInteraction
plot(bf)

# step 2 -There isn't! Let's colapse across setsize then. 

bf = anovaBF(Accuracy.... ~ NFTraining * Testday + SubID , data=behavedata, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[5] # is the full model better than the attention train knock out?
bf_main_testday = bf[7]/bf[6] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_testday
plot(bf)

# Step 3 - there's an interaction between testday and NF training, lets check how training effects compared across training
bf = anovaBF(AccuracyTrEfct ~ NFTraining , data=behavedata_train, whichRandom="subID")
bf
plot(bf)


datuse=behavedata[behavedata['NFTraining']=='Neurofeedback',]
tapply(datuse$Accuracy....,datuse$Testday, mean)
bf = anovaBF(Accuracy.... ~ Testday  + SubID , data=datuse, whichRandom="SubID",  whichModels="all")
bf
plot(bf)

datuse=behavedata[behavedata['NFTraining']=='Sham',]
tapply(datuse$Accuracy....,datuse$Testday, mean)
bf = anovaBF(Accuracy.... ~ Testday  + SubID , data=datuse, whichRandom="SubID",  whichModels="all")
bf
plot(bf)


#hmm, the neurofeedback group got worse, but the sham group stayed the same!
# todo here - bring in some pre-vs post ttests and descritive stats. functionalise. 




## RT
# step 1 - check if set size interacts with training effects
## Test effects of NF, with set size considered

bf = anovaBF(RT_TrEfct ~ Attention.Trained * Set.Size + SubID , data=behavedata_train, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?

# display results
bf
bfInteraction
plot(bf)


# step 2 -There isn't! Let's colapse across setsize then. 


bf = anovaBF(Reaction.Time..s. ~ NFTraining * Testday + SubID , data=behavedata, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[5] # is the full model better than the attention train knock out?
bf_main_testday = bf[7]/bf[6] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_testday
plot(bf)


