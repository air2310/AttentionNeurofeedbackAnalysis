library(dplyr)
library(BayesFactor)
library(car)

#reference for Bayesian Anova method applied here: Rouder et al 2012.
# https://www.sciencedirect.com/science/article/pii/S0022249612000806?casa_token=7qVp7duocU0AAAAA:isxnVW-Oi6-62_oaPg-PJx1oPB4uNliU9GsP__W_bAOFozswKTky_VH9mq94omHGQb9yWyUKxU7g

# load data
behavedata = read.csv("//data.qbi.uq.edu.au/VISATTNNF-Q1357/Results/CompareSpaceFeat/group/motiondiscrim_behaveresults_ALL.csv")


#### create factors

behavedata_orig = behavedata

behavedata$AttentionTrained = factor(behavedata$AttentionTrained)
levels(behavedata$AttentionTrained) = c( "Feature", "Sham","Space")

behavedata$NFTraining = behavedata$AttentionTrained
levels(behavedata$NFTraining) = c( "Neurofeedback", "Sham","Neurofeedback")

behavedata$Attention.Type = factor(behavedata$Attention.Type)
levels(behavedata$Attention.Type) = c( "Feature", "Space")

behavedata$Testday = factor(behavedata$Testday)
levels(behavedata$Testday) = c( "pre-training", "post-training")

behavedata$subID = factor(behavedata$subID)

#### Calculate training effects

idx_d1 =behavedata$Testday=="pre-training"
idx_d4 =behavedata$Testday=="post-training"

tmp_d1 = behavedata[idx_d1,]
tmp_d4 = behavedata[idx_d4,]

behavedata_train = tmp_d1
behavedata_train$Sensitivity_TrEfct = tmp_d4$Sensitivity - tmp_d1$Sensitivity
behavedata_train$Criterion_TrEfct = tmp_d4$Criterion - tmp_d1$Criterion
behavedata_train$LikelihoodRatio_TrEfct = tmp_d4$LikelihoodRatio - tmp_d1$LikelihoodRatio
behavedata_train$correct_TrEfct = tmp_d4$correct - tmp_d1$correct

behavedata_train$RT_TrEfct = tmp_d4$RT - tmp_d1$RT
behavedata_train$RT_STD_TrEfct = tmp_d4$RT_STD - tmp_d1$RT_STD
behavedata_train$IE_TrEfct = tmp_d4$InverseEfficiency - tmp_d1$InverseEfficiency


##### define functions to display 2x2 BF ANOVA results and followups
DisplayANOVAResults2X2 <- function(formula, data) {
  bf = anovaBF(formula, data=data, whichRandom="subID",  whichModels="all")
  bfInteraction = bf[7]/bf[4] # is the full model better than just strait main effects?
  bf_main_A = bf[7]/bf[5] # is the full model better than the main effect A knock out?
  bf_main_B = bf[7]/bf[6] # is the full model better than the main effect B knock out?
  
  # display results
  print(bf)
  print('Interaction:')
  print(bfInteraction)
  
  print('Main effect A:')
  print(bf_main_A)
  
  print('Main effect B:')
  print(bf_main_B)
  
  plot(bf)
}

followuptests <- function(formula, dataA, dataB) {
  bf = ttestBF(formula = formula, data = dataA)
  print('Data A')
  print(bf)
  print('Data B')
  bf = ttestBF(formula = formula, data = dataB)
  print(bf)
}


Runstats_NFvsSham <- function(datuse, datuse_train, datuse_NF, datuse_sham) {
  
  print('Sensitivity')
  DisplayANOVAResults2X2(Sensitivity ~ NFTraining * Testday + subID , data=datuse)
  followuptests(formula = Sensitivity ~  Testday, dataA = datuse_NF, dataB = datuse_sham)
  
  t.test(Sensitivity_TrEfct ~  NFTraining, data = datuse_train, var.eq=TRUE)
  bf = ttestBF(formula =Sensitivity_TrEfct ~  NFTraining, data = datuse_train)
  
  print('Reaction Time')
  DisplayANOVAResults2X2(formula = RT ~ NFTraining * Testday + subID , data=datuse)
  followuptests(formula = RT ~  Testday, dataA = datuse_NF, dataB = datuse_sham)
  
  print('% Correct')
  DisplayANOVAResults2X2(correct ~ NFTraining * Testday + subID , data=datuse)
  followuptests(formula = correct ~  Testday, dataA = datuse_NF, dataB = datuse_sham)
  
  print('Criterion')
  DisplayANOVAResults2X2(Criterion ~ NFTraining * Testday + subID , data=datuse)
  followuptests(formula = Criterion ~  Testday, dataA = datuse_NF, dataB = datuse_sham)
  
  print('Inverse Efficiency')
  DisplayANOVAResults2X2(InverseEfficiency ~ NFTraining * Testday + subID , data=datuse)
  followuptests(formula = InverseEfficiency ~  Testday, dataA = datuse_NF, dataB = datuse_sham)
  
}


Runstats_SpaceVsFeat <- function(datuse_train) {
  
  print('Sensitivity')
  DisplayANOVAResults2X2(Sensitivity_TrEfct ~ AttentionTrained * Attention.Type + subID , data=datuse_train)
  
  print('Reaction Time')
  DisplayANOVAResults2X2(RT_TrEfct ~ AttentionTrained * Attention.Type + subID , data=datuse_train)
  
  print('% Correct')
  DisplayANOVAResults2X2(correct_TrEfct ~ AttentionTrained * Attention.Type + subID , data=datuse_train)
  
  print('Criterion')
  DisplayANOVAResults2X2(Criterion_TrEfct ~ AttentionTrained * Attention.Type + subID , data=datuse_train)

  print('Inverse Efficiency')
  DisplayANOVAResults2X2(IE_TrEfct ~ AttentionTrained * Attention.Type + subID , data=datuse_train)
  
}

############# First we'll assess whether or not Neurofeedback groups differed from Sham. #############
#### lets start with the space task
taskstring = 'Space'
datuse = behavedata[behavedata['Attention.Type']==taskstring,]
datuse_train = behavedata_train[behavedata_train['Attention.Type']==taskstring,]
datuse_NF = datuse[datuse['NFTraining']=="Neurofeedback",]
datuse_sham = datuse [datuse ['NFTraining']=="Sham",]

Runstats_NFvsSham(datuse, datuse_train, datuse_NF, datuse_sham)

#### What about the Feature Task?
taskstring = 'Feature'
datuse = behavedata[behavedata['Attention.Type']==taskstring,]
datuse_train = behavedata_train[behavedata_train['Attention.Type']==taskstring,]
datuse_NF = datuse[datuse['NFTraining']=="Neurofeedback",]
datuse_sham = datuse [datuse ['NFTraining']=="Sham",]

Runstats_NFvsSham(datuse, datuse_train, datuse_NF, datuse_sham)


############# Next up, What about the effect of the specific type of neurofeedack training? #############

datuse_train = behavedata_train[ behavedata_train['AttentionTrained']!="Sham",]
Runstats_SpaceVsFeat(datuse_train)


# Ancova
#fit2=aov(Sensitivity_TrEfct~ NFTraining + Sensitivity ,datuse)
#Anova(fit2, type="III")