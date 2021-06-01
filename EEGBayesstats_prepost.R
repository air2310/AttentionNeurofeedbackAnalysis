library(BayesFactor)
library(R.matlab)
library(ggplot2)
library(RcppCNPy)

library(reshape2)
library(plyr)
library(RColorBrewer)
library(ggpubr)
library(ggthemes)
library(tidyverse)

# Settings ----------------------------------------------------------------

# run options???

options.traintype =3# Feature or Space or SHAM
options.singletrial =2# ERP, single trial

# task options
options.task = 1 # 1 = Motion Detection,
options.TestTrain = 1 # 1 = test, 2 = train

# analysis options
n_days = switch (options.TestTrain, 2, 3)

# colors
paletteuse = "BrBG" #"PRGn"

yellow = rgb(242 / 255, 176 / 255, 53 / 255)
orange = rgb(236 / 255, 85 / 255, 58 / 255)
darkteal = rgb(18 / 255, 47 / 255, 65 / 255)
medteal = rgb(5 / 255, 133 / 255, 134 / 255)
lightteal = rgb(78 / 255, 185 / 255, 159 / 255)

# strings
strings.day = c("day-1", "day-4")
strings.task = c("AttnNFMotion",  "AttnNFVisualSearch");
strings.TestTrain = c("Test", "Train");
strings.traintype = c("Feature", "Space",)
strings.singletrial = c("", "_epochs")

# Directories -------------------------------------------------------------

# set root directories
direct.Root = "//data.qbi.uq.edu.au/VISATTNNF-Q1357/"
direct.resultsRoot = paste(direct.Root, "Results/", sep = "")
direct.resultsGroup = paste(direct.Root, "Results/group", sep = "")
direct.resultsGroup = paste(direct.resultsRoot, "Train",strings.traintype[options.traintype], "/group",sep = "")

# set filenames

filename.bids.results = paste( "group_ssvep_selectivity_prepost", strings.singletrial[options.singletrial], ".npy", sep = "") # day 1-space, day 1 - feature, day 4 - space, day 4 - feature
results <- npyLoad( paste( direct.resultsGroup, "/",  filename.bids.results, sep = ""))

#results = results[,sample(0:23,23, replace = FALSE)]


# Assign to dataframe: day 1-space, day 1 - feature, day 4 - space, day 4 - feature

tmp1 = data.frame(ssveps = results[1,])
tmp1["day"] <- "pre-training"
tmp1["attn"] <- "space"

tmp2 = data.frame(ssveps = results[2,])
tmp2["day"] <- "pre-training"
tmp2["attn"] <- "feature"

tmp3 = data.frame(ssveps = results[3,])
tmp3["day"] <- "post-training"
tmp3["attn"] <- "space"

tmp4 = data.frame(ssveps = results[4,])
tmp4["day"] <- "post-training"
tmp4["attn"] <- "feature"

selectivity = rbind(tmp1, tmp2, tmp3, tmp4)

# Get Data ----------------------------------------------------------------

#file.ls = list.files(path=direct.resultsGroup, pattern=glob2rx(paste( filename.bids.results, "*.npy", sep = "")))
#
#DATA_ALL = readMat(paste(direct.results, tail(file.ls, n=1), sep = ""))

# assign dat
#SELECTIVITY = DATA_ALL$Selectivity.ALL

## Preallocate bayes factors
n.subs = length( results[1,])
BAYESPLOT = data.frame()

## Compute bayes factors

for (SS in 2:n.subs) {

  # space comparison
  statdat = subset(selectivity, attn=="space")
  dat1 = subset(statdat,day=="pre-training")
  dat2=subset(statdat,day=="post-training")

  bf = ttestBF(dat1$ssveps[0:SS], dat2$ssveps[0:SS], paired = TRUE)
  BF1=data.frame(bf, stringsAsFactors = FASLE)

  # feature comparison
  statdat = subset(selectivity, attn=="feature")
  dat1 = subset(statdat,day=="pre-training")
  dat2=subset(statdat,day=="post-training")

  bf = ttestBF(dat1$ssveps[0:SS], dat2$ssveps[0:SS], paired = TRUE)
  BF2=data.frame(bf, stringsAsFactors = FASLE)

  tmp = cbind(data.frame(BF_space = BF1$bf), data.frame(BF_feature = BF2$bf), data.frame(subjects =SS))
  BAYESPLOT = rbind(BAYESPLOT, tmp)
}

## invert
#BAYESPLOT['BF_space'] = 1/BAYESPLOT['BF_space']
#BAYESPLOT['BF_feature'] = 1/BAYESPLOT['BF_feature']

## plot

plotdat = melt(BAYESPLOT, id = "subjects" )

i <- ggplot(plotdat , aes( x=subjects, y=value, color=variable))
i <- i + geom_line() + geom_point()

tit = paste("Train ", strings.traintype[options.traintype], " Bayes Factor by Subject Number", strings.singletrial[options.singletrial])
i <- i + ggtitle( tit ) + theme(plot.title = element_text(hjust = 0.5))

i <- i + scale_y_continuous(trans='log2')
i <- i + geom_hline(yintercept=3, linetype="dashed", color = yellow)
i <- i + geom_hline(yintercept=0.33, linetype="dashed", color = yellow)
i <- i + geom_vline(xintercept=30, linetype="dashed", color = orange)

i <- i + scale_colour_manual(values = c(darkteal, lightteal) )
plot(i)
ggsave(paste(direct.resultsGroup, "/", tit, "H0.png", sep=""))

