library(dplyr)
library(BayesFactor)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
behavedata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/motiondiscrim_behaveresults_ALL_duringNF.csv")


#### Step 2 - create factors

behavedata_orig = behavedata

behavedata$AttentionTrained = factor(behavedata$AttentionTrained)
levels(behavedata$AttentionTrained) = c( "Feature", "Space")

behavedata$Testday = factor(behavedata$Testday)
levels(behavedata$Testday) = c( "Day 1", "Day 2", "Day 3")

behavedata$subID = factor(behavedata$subID)

behavedata_RT = na.omit(behavedata)


#### Step 2 - Assess whether Training effect interacts with training group. 

### sensitivity
bf = anovaBF(Sensitivity ~ AttentionTrained * Testday + subID , data=behavedata, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_Testday = bf[7]/bf[5] # is the full model better than the Testday knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_Testday
plot(bf)



### Criterion
bf = anovaBF(Criterion ~ AttentionTrained * Testday + subID , data=behavedata, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_Testday = bf[7]/bf[5] # is the full model better than the Testday knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_Testday
plot(bf)



### LikelihoodRatio
bf = anovaBF(LikelihoodRatio ~ AttentionTrained * Testday + subID , data=behavedata, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_Testday = bf[7]/bf[5] # is the full model better than the Testday knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_Testday
plot(bf)


### Correct
bf = anovaBF(correct ~ AttentionTrained * Testday + subID , data=behavedata, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_Testday = bf[7]/bf[5] # is the full model better than the Testday knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_Testday
plot(bf)


### RT
bf = anovaBF(RT ~ AttentionTrained * Testday + subID , data=behavedata_RT, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_Testday = bf[7]/bf[5] # is the full model better than the Testday knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_Testday
plot(bf)


