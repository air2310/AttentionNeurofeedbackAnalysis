library(dplyr)
library(BayesFactor)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
behavedata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/motiondiscrim_behaveresults_ALL.csv")

#### step 1 - visualise the data 
coplot(Sensitivity ~ AttentionTrained |  Testday, data = behavedata, panel = panel.smooth,
       xlab = "Sensitivity data: Sensitivity vs attention trained, given testday")

#### Step 2 - create factors

behavedata_orig = behavedata

behavedata$AttentionTrained = factor(behavedata$AttentionTrained)
levels(behavedata$AttentionTrained) = c( "Feature", "Space")

behavedata$Attention.Type = factor(behavedata$Attention.Type)
levels(behavedata$Attention.Type) = c( "Feature", "Space")

behavedata$Testday = factor(behavedata$Testday)
levels(behavedata$Testday) = c( "Day 1", "Day 4")

behavedata$subID = factor(behavedata$subID)

behavedata_RT = na.omit(behavedata)

#### Step 3 - was there an overall training effect? i.e. did people improve overall after training. 

bf = anovaBF(Sensitivity ~ Testday + subID , data=behavedata, whichRandom="subID")
bf #  Print to console
plot(bf) # plot

bf = anovaBF(Criterion ~ Testday + subID , data=behavedata, whichRandom="subID")
bf #  Print to console
plot(bf) # plot

bf = anovaBF(LikelihoodRatio ~ Testday + subID , data=behavedata, whichRandom="subID")
bf #  Print to console
plot(bf) # plot

bf = anovaBF(correct ~ Testday + subID , data=behavedata, whichRandom="subID")
bf #  Print to console
plot(bf) # plot

bf = anovaBF(RT ~ Testday + subID , data=behavedata_RT, whichRandom="subID")
bf #  Print to console
plot(bf) # plot

# Were there overall effects of training group?

bf = anovaBF(Sensitivity ~ AttentionTrained + subID , data=behavedata, whichRandom="subID")
bf #  Print to console
plot(bf) # plot

bf = anovaBF(Criterion ~ AttentionTrained + Attention.Type + subID , data=behavedata[behavedata$Testday=='Day 4',], whichRandom="subID")
bf #  Print to console
plot(bf) # plot


bf = anovaBF(RT ~ AttentionTrained + subID , data=behavedata_RT, whichRandom="subID")
bf #  Print to console
plot(bf) # plot

#### Step 4 - Create Training effect vector (sensitivity on day 1 vs. day 4)

idx_d1 =behavedata$Testday=="Day 1"
idx_d4 =behavedata$Testday=="Day 4"

tmp_d1 = behavedata[idx_d1,]
tmp_d4 = behavedata[idx_d4,]

behavedata_train = tmp_d1[,0:4]
behavedata_train$Sensitivity_TrEfct = tmp_d4$Sensitivity - tmp_d1$Sensitivity
behavedata_train$Criterion_TrEfct = tmp_d4$Criterion - tmp_d1$Criterion
behavedata_train$LikelihoodRatio_TrEfct = tmp_d4$LikelihoodRatio - tmp_d1$LikelihoodRatio
behavedata_train$correct_TrEfct = tmp_d4$correct - tmp_d1$correct
behavedata_train$RT_TrEfct = tmp_d4$RT - tmp_d1$RT

behavedata_RT_train = na.omit(behavedata_train)

#### Step 5 - Assess whether this training effect differs across the levels of Attention Trained and attention type

# what is the best model?
# Test for Interaction - is the full model better than the full model with interaction knocked out?
# test for main effects - is the full model better than the full model with the main effect of the factor knocked out? 

### sensitivity
bf = anovaBF(Sensitivity_TrEfct ~ AttentionTrained * Attention.Type + subID , data=behavedata_train, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_AttnType = bf[7]/bf[5] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_AttnType
plot(bf)



### Criterion
bf = anovaBF(Criterion_TrEfct ~ AttentionTrained * Attention.Type + subID , data=behavedata_train, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_AttnType = bf[7]/bf[5] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_AttnType
plot(bf)



### LikelihoodRatio
bf = anovaBF(LikelihoodRatio_TrEfct ~ AttentionTrained * Attention.Type + subID , data=behavedata_train, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_AttnType = bf[7]/bf[5] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_AttnType
plot(bf)


### Correct
bf = anovaBF(correct_TrEfct ~ AttentionTrained * Attention.Type + subID , data=behavedata_train, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_AttnType = bf[7]/bf[5] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_AttnType
plot(bf)


### RT
bf = anovaBF(RT_TrEfct ~ AttentionTrained * Attention.Type + subID , data=behavedata_RT_train, whichRandom="subID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_AttnType = bf[7]/bf[5] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_AttnType
plot(bf)


# 
# # old stuff
# # summarise our continuous data
# continuous <-select_if(behavedata, is.numeric)
# summary(continuous)
# 
# 
# # summarise our factors
# factor <- data.frame(select_if(behavedata, is.factor))
# ncol(factor)
# 
# # run regular anovas
# anova <- aov(Sensitivity ~ AttentionTrained * Attention.Type * Testday , behavedata)
# summary(anova)
# 
# anova <- aov(correct ~ AttentionTrained * Attention.Type * Testday , behavedata)
# summary(anova)
# 
# anova <- aov(Criterion ~ AttentionTrained * Attention.Type * Testday , behavedata)
# summary(anova)
# 
# # Bayes Anova - plot
# coplot(Sensitivity ~ AttentionTrained |  Testday, data = behavedata, panel = panel.smooth,
#        xlab = "Sensitivity data: Sensitivity vs attention trained, given testday")
# 
# # Bayes Anova - factorise
# behavedata$AttentionTrained = factor(behavedata$AttentionTrained)
# levels(behavedata$AttentionTrained) = c( "Feature", "Space")
# 
# behavedata$Attention.Type = factor(behavedata$Attention.Type)
# levels(behavedata$Attention.Type) = c( "Feature", "Space")
# 
# behavedata$Testday = factor(behavedata$Testday)
# levels(behavedata$Testday) = c( "Day 1", "Day 4")
# 
# behavedata$subID = factor(behavedata$subID)
# 
# # Bayes Anova - get that stat!
# #bf = anovaBF(Sensitivity ~ AttentionTrained * Attention.Type * Testday , data=behavedata)
# #bf
# 
# #plot(bf[8] / bf[18])
# 
# #bfMainEffects = lmBF(Sensitivity ~ Testday, data = behavedata)
# #bfMainEffects2 = lmBF(Sensitivity ~ Attention.Type + Testday, data = behavedata)
# #bfInteraction = lmBF(Sensitivity ~ AttentionTrained + Attention.Type + Testday + AttentionTrained : Attention.Type : Testday, data = behavedata)
# 
# 
# bf = anovaBF(Sensitivity ~ AttentionTrained * Attention.Type * Testday + subID , data=behavedata, whichRandom="subID")
# bf
# 
# plot(bf)
# 
# 
# #model = lmBF(Sensitivity ~ Testday, data = behavedata)
# #bfMainEffects2 = lmBF(Sensitivity ~ Attention.Type + Testday, data = behavedata)
# #bfInteraction = lmBF(Sensitivity ~ AttentionTrained + Attention.Type + Testday + AttentionTrained : Attention.Type : Testday, data = behavedata)
# 
# # plan - list all the models I want including the 3 way interaction, then compare all these to the SUBID only one. 


