library(dplyr)
library(BayesFactor)
library(car)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
SSVEPdata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/motiondiscrim_SelectivityResults_ALL.csv")
SSVEPdata$SSVEP = SSVEPdata$Selectivity..Î.ÂµV.

#### step 1 - visualise the data 
coplot(SSVEP ~ Attention.Trained |  Testday, data = SSVEPdata, panel = panel.smooth,
       xlab = "Selectivity data: Selectivity vs attention trained, given testday")

#### Step 2 - create factors

SSVEPdata_orig = SSVEPdata

SSVEPdata$Attention.Trained = factor(SSVEPdata$Attention.Trained)
levels(SSVEPdata$Attention.Trained) = c( "Feature", "Space")

SSVEPdata$Attention.Type = factor(SSVEPdata$Attention.Type)
levels(SSVEPdata$Attention.Type) = c( "Feature", "Space")

SSVEPdata$Testday = factor(SSVEPdata$Testday)
levels(SSVEPdata$Testday) = c( "post-training", "pre-training")

SSVEPdata$SubID = factor(SSVEPdata$SubID)


#### Step 3 - was there an overall training effect? i.e. did people improve overall after training. 

bf = anovaBF(SSVEP ~ Testday + SubID , data=SSVEPdata, whichRandom="SubID")
bf #  Print to console
plot(bf) # plot

# Bayes factor analysis
# --------------
#   [1] Testday + SubID : 0.1278655 ±0.81%
# 
# Against denominator:
#   SSVEP ~ SubID 
# ---
#   Bayes factor type: BFlinearModel, JZS
 

#### Step 4 - Create Training effect vector (sensitivity on day 1 vs. day 4)

idx_d1 = SSVEPdata$Testday=="pre-training"
idx_d4 = SSVEPdata$Testday=="post-training"

tmp_d1 = SSVEPdata[idx_d1,]
tmp_d4 = SSVEPdata[idx_d4,]

SSVEPdata_train = tmp_d1[,0:4]
SSVEPdata_train$SSVEP_TrEfct = tmp_d4$SSVEP - tmp_d1$SSVEP

#### Step 5 - Assess whether this training effect differs across the levels of Attention Trained and attention type

# what is the best model?
# Test for Interaction - is the full model better than the full model with interaction knocked out?
# test for main effects - is the full model better than the full model with the main effect of the factor knocked out? 


bf = anovaBF(SSVEP_TrEfct  ~ Attention.Trained * Attention.Type + SubID , data=SSVEPdata_train, whichRandom="SubID",  whichModels="all")
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_AttnTrain = bf[7]/bf[6] # is the full model better than the attention train knock out?
bf_main_AttnType = bf[7]/bf[5] # is the full model better than the attention type knock out?

# display results
bf
bfInteraction
bf_main_AttnTrain
bf_main_AttnType
plot(bf)

# > bfInteraction
# Bayes factor analysis
# --------------
#   [1] Attention.Type + Attention.Trained + Attention.Type:Attention.Trained + SubID : 0.258135 ±3.57%
# 
# Against denominator:
#   SSVEP_TrEfct ~ Attention.Type + Attention.Trained + SubID 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 
# > bf_main_AttnTrain
# Bayes factor analysis
# --------------
#   [1] Attention.Type + Attention.Trained + Attention.Type:Attention.Trained + SubID : 0.1867183 ±3.6%
# 
# Against denominator:
#   SSVEP_TrEfct ~ Attention.Trained + Attention.Type:Attention.Trained + SubID 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 
# > bf_main_AttnType
# Bayes factor analysis
# --------------
#   [1] Attention.Type + Attention.Trained + Attention.Type:Attention.Trained + SubID : 0.2165626 ±2.51%
# 
# Against denominator:
#   SSVEP_TrEfct ~ Attention.Type + Attention.Type:Attention.Trained + SubID 
# ---
#   Bayes factor type: BFlinearModel, JZS
