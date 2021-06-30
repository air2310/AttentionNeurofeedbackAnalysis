library(dplyr)
library(BayesFactor)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
RSAdata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/RSA_results.csv")

#### create factors

RSAdata_orig = RSAdata

RSAdata$TrainingGroup = factor(RSAdata$TrainingGroup)

RSAdata$NFTraining = RSAdata$TrainingGroup
levels(RSAdata$NFTraining) = c( "Neurofeedback", "Sham","Neurofeedback")

RSAdata$Bootstrap = factor(RSAdata$Bootstrap)


## TTests

print(ttestBF(formula = RDM_Score ~ NFTraining, data=RSAdata))

datuse = RSAdata[RSAdata$TrainingGroup == "Space" | RSAdata$TrainingGroup == "Sham",]
datuse$TrainingGroup = factor(datuse$TrainingGroup)
print(ttestBF(formula = RDM_Score ~ TrainingGroup, data=datuse))


datuse = RSAdata[RSAdata$TrainingGroup == "Feature" | RSAdata$TrainingGroup == "Sham",]
datuse$TrainingGroup = factor(datuse$TrainingGroup)
print(ttestBF(formula = RDM_Score ~ TrainingGroup, data=datuse))


datuse = RSAdata[RSAdata$TrainingGroup == "Space" | RSAdata$TrainingGroup == "Feature",]
datuse$TrainingGroup = factor(datuse$TrainingGroup)
print(ttestBF(formula = RDM_Score ~ TrainingGroup, data=datuse))

bf = anovaBF(RDM_Score ~ TrainingGroup , data=RSAdata, whichRandom="bootstrap",  whichModels="all")
print(bf)

# print(ttestBF(formula = RDM_Score ~ NFTraining, data=RSAdata))
# t is large; approximation invoked.
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 1.939618e+38 ±0%
# 
# Against denominator:
#   Null, mu1-mu2 = 0 
# ---
#   Bayes factor type: BFindepSample, JZS
# 
# > datuse = RSAdata[RSAdata$TrainingGroup == "Space" | RSAdata$TrainingGroup == "Sham",]
# > datuse$TrainingGroup = factor(datuse$TrainingGroup)
# > print(ttestBF(formula = RDM_Score ~ TrainingGroup, data=datuse))
# t is large; approximation invoked.
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 2.721014e+51 ±0%
# 
# Against denominator:
#   Null, mu1-mu2 = 0 
# ---
#   Bayes factor type: BFindepSample, JZS
# 
# > datuse = RSAdata[RSAdata$TrainingGroup == "Feature" | RSAdata$TrainingGroup == "Sham",]
# > datuse$TrainingGroup = factor(datuse$TrainingGroup)
# > print(ttestBF(formula = RDM_Score ~ TrainingGroup, data=datuse))
# t is large; approximation invoked.
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 6.797265e+34 ±0%
# 
# Against denominator:
#   Null, mu1-mu2 = 0 
# ---
#   Bayes factor type: BFindepSample, JZS
# 
# > 
#   > 
#   > datuse = RSAdata[RSAdata$TrainingGroup == "Space" | RSAdata$TrainingGroup == "Feature",]
# > datuse$TrainingGroup = factor(datuse$TrainingGroup)
# > print(ttestBF(formula = RDM_Score ~ TrainingGroup, data=datuse))
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 2.040349e+22 ±0%
# 
# Against denominator:
#   Null, mu1-mu2 = 0 
# ---
#   Bayes factor type: BFindepSample, JZS
# 
# > 
#   > bf = anovaBF(RDM_Score ~ TrainingGroup , data=RSAdata, whichRandom="bootstrap",  whichModels="all")
# |======================================================================================================================================================| 100%
# > print(bf)
# Bayes factor analysis
# --------------
#   [1] TrainingGroup : 9.694998e+68 ±0%
# 
# Against denominator:
#   Intercept only 
# ---
#   Bayes factor type: BFlinearModel, JZS

## By Behave


# load data
RSAdata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/RSA_bybehave_results.csv")

#### create factors

RSAdata_orig = RSAdata

RSAdata$TrainingGroup = factor(RSAdata$TrainingGroup)
RSAdata$Cuetype = factor(RSAdata$Cuetype)


## Stats
bf = anovaBF(RDM_Score_effect ~ TrainingGroup*Cuetype , data=RSAdata,  whichModels="all")
print(bf)
bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
bf_main_A = bf[7]/bf[5] # is the full model better than the main effect A knock out?
bf_main_B = bf[7]/bf[6] # is the full model better than the main effect B knock out?

print(bfInteraction)
print(bf_main_A)
print(bf_main_B)


datuse = RSAdata[RSAdata$TrainingGroup == "Space",]
print(ttestBF(formula = RDM_Score_effect ~ Cuetype, data=datuse))


datuse = RSAdata[RSAdata$TrainingGroup == "Feature",]
print(ttestBF(formula = RDM_Score_effect ~ Cuetype, data=datuse))


ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Feature" & RSAdata$Cuetype == "Feature Task","RDM_Score_effect"])
ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Feature" & RSAdata$Cuetype == "Space Task","RDM_Score_effect"])

ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Space" & RSAdata$Cuetype == "Feature Task","RDM_Score_effect"])
ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Space" & RSAdata$Cuetype == "Space Task","RDM_Score_effect"])

# ## Stats
# > bf = anovaBF(RDM_Score_effect ~ TrainingGroup*Cuetype , data=RSAdata,  whichModels="all")
# |===========================================                                                                                                           |  29%t is large; approximation invoked.
# |======================================================================================================================================================| 100%
# > print(bf)
# Bayes factor analysis
# --------------
#   [1] TrainingGroup                                   : 0.1182574    ±0%
# [2] Cuetype                                         : 3501.62      ±0%
# [3] TrainingGroup:Cuetype                           : 2.797604e+79 ±0%
# [4] TrainingGroup + Cuetype                         : 417.0372     ±0.97%
# [5] TrainingGroup + TrainingGroup:Cuetype           : 2.662753e+78 ±1.9%
# [6] Cuetype + TrainingGroup:Cuetype                 : 2.591524e+90 ±2.65%
# [7] TrainingGroup + Cuetype + TrainingGroup:Cuetype : 3.27831e+89  ±1.88%
# 
# Against denominator:
#   Intercept only 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 
# > bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
# > bf_main_A = bf[7]/bf[5] # is the full model better than the main effect A knock out?
# > bf_main_B = bf[7]/bf[6] # is the full model better than the main effect B knock out?
# > 
#   > print(bfInteraction)
# Bayes factor analysis
# --------------
#   [1] TrainingGroup + Cuetype + TrainingGroup:Cuetype : 7.860955e+86 ±2.12%
# 
# Against denominator:
#   RDM_Score_effect ~ TrainingGroup + Cuetype 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 
# > print(bf_main_A)
# Bayes factor analysis
# --------------
#   [1] TrainingGroup + Cuetype + TrainingGroup:Cuetype : 123117351786 ±2.68%
# 
# Against denominator:
#   RDM_Score_effect ~ TrainingGroup + TrainingGroup:Cuetype 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 
# > print(bf_main_B)
# Bayes factor analysis
# --------------
#   [1] TrainingGroup + Cuetype + TrainingGroup:Cuetype : 0.1265013 ±3.24%
# 
# Against denominator:
#   RDM_Score_effect ~ Cuetype + TrainingGroup:Cuetype 
# ---
#   Bayes factor type: BFlinearModel, JZS
# 
# > 
#   > 
#   > datuse = RSAdata[RSAdata$TrainingGroup == "Space",]
# > print(ttestBF(formula = RDM_Score_effect ~ Cuetype, data=datuse))
# t is large; approximation invoked.
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 6.651202e+46 ±0%
# 
# Against denominator:
#   Null, mu1-mu2 = 0 
# ---
#   Bayes factor type: BFindepSample, JZS
# 
# > 
#   > 
#   > datuse = RSAdata[RSAdata$TrainingGroup == "Feature",]
# > print(ttestBF(formula = RDM_Score_effect ~ Cuetype, data=datuse))
# t is large; approximation invoked.
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 3.083495e+39 ±0%
# 
# Against denominator:
#   Null, mu1-mu2 = 0 
# ---
#   Bayes factor type: BFindepSample, JZS
# 
# > 
#   > 
#   > ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Feature" & RSAdata$Cuetype == "Feature Task","RDM_Score_effect"])
# t is large; approximation invoked.
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 4.57594e+31 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Feature" & RSAdata$Cuetype == "Space Task","RDM_Score_effect"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 0.4821255 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > 
#   > ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Space" & RSAdata$Cuetype == "Feature Task","RDM_Score_effect"])
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 54078886502 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS
# 
# > ttestBF(x=RSAdata[RSAdata$TrainingGroup == "Space" & RSAdata$Cuetype == "Space Task","RDM_Score_effect"])
# t is large; approximation invoked.
# Bayes factor analysis
# --------------
#   [1] Alt., r=0.707 : 1.393733e+32 ±0%
# 
# Against denominator:
#   Null, mu = 0 
# ---
#   Bayes factor type: BFoneSample, JZS