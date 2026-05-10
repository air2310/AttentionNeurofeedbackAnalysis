library(dplyr)
library(BayesFactor)
library(car)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
behavedata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/NBACK_behaveresults_ALL.csv")

#### Step 2 - create factors

behavedata_orig = behavedata

behavedata$Attention.Trained = factor(behavedata$Attention.Trained)
levels(behavedata$Attention.Trained) = c( "Feature", "Sham","Space")

behavedata$NFTraining = behavedata$Attention.Trained
levels(behavedata$NFTraining) = c( "Neurofeedback", "Sham","Neurofeedback")


behavedata$Testday = factor(behavedata$Testday)
levels(behavedata$Testday) = c("post-training", "pre-training")

behavedata$SubID = factor(behavedata$SubID)

behavedata_RT = na.omit(behavedata)

## Test effects of NF

bf = anovaBF(Sensitivity ~ Attention.Trained * Testday + SubID , data=behavedata, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_testday = bf[7]/bf[5] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_testday
plot(bf)


bf = anovaBF(Reaction.Time..s. ~ Attention.Trained * Testday + SubID , data=behavedata, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[5] # is the full model better than the attention train knock out?
bf_main_testday = bf[7]/bf[6] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_testday
plot(bf)


bf = anovaBF(LikelihoodRatio ~ Attention.Trained * Testday + SubID , data=behavedata, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[5] # is the full model better than the attention train knock out?
bf_main_testday = bf[7]/bf[6] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_testday
plot(bf)

# Follow up interaction by looking at how training effects differ by training type
bf = anovaBF(Sensitivity_TrEfct ~ NFTraining , data=behavedata, whichRandom="subID")
bf
plot(bf)

# Ancova
fit2=aov(Sensitivity_TrEfct~ AttentionTrained + Sensitivity_pre ,datuse)
Anova(fit2, type="III")

