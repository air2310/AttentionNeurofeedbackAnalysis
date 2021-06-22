library(dplyr)
library(BayesFactor)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
behavedata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/motiondiscrim_behaveresults_ALL_duringNF.csv")

#### create factors

behavedata_orig = behavedata

behavedata$AttentionTrained = factor(behavedata$AttentionTrained)
levels(behavedata$AttentionTrained) = c( "Feature", "Sham","Space")

behavedata$NFTraining = behavedata$AttentionTrained
levels(behavedata$NFTraining) = c( "Neurofeedback", "Sham","Neurofeedback")

behavedata$Testday = factor(behavedata$Testday)
levels(behavedata$Testday) = c( "Day 1", "Day 2", "Day 3")

behavedata$subID = factor(behavedata$subID)

behavedata_RT = na.omit(behavedata)


#### Calculate training effects

idx_d1 =behavedata$Testday=="Day 1"
idx_d3 =behavedata$Testday=="Day 3"

tmp_d1 = behavedata[idx_d1,]
tmp_d3 = behavedata[idx_d3,]

behavedata_train = tmp_d1
behavedata_train$Sensitivity_TrEfct     = (tmp_d3$Sensitivity - tmp_d1$Sensitivity) / 2
behavedata_train$Criterion_TrEfct       = (tmp_d3$Criterion - tmp_d1$Criterion)/ 2
behavedata_train$LikelihoodRatio_TrEfct = (tmp_d3$LikelihoodRatio - tmp_d1$LikelihoodRatio)/ 2
behavedata_train$correct_TrEfct         = (tmp_d3$correct - tmp_d1$correct)/ 2
behavedata_train$RT_TrEfct              = (tmp_d3$RT - tmp_d1$RT)/ 2

## Test!ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Space","RT_TrEfct"])
ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Space","RT_TrEfct"])
ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Feature","RT_TrEfct"])
ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Sham","RT_TrEfct"])

ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Space","Sensitivity_TrEfct"])
ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Feature","Sensitivity_TrEfct"])
ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Sham","Sensitivity_TrEfct"])


# ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Space","RT_TrEfct"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 215.4644 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Feature","RT_TrEfct"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 3.488154 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Sham","RT_TrEfct"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 0.258749 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > 
#   > ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Space","Sensitivity_TrEfct"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 3.516298 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Feature","Sensitivity_TrEfct"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 0.4731049 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > ttestBF(x=behavedata_train[behavedata_train['AttentionTrained']=="Sham","Sensitivity_TrEfct"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 0.5999819 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS














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


