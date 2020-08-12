library(BayesFactor)
library(R.matlab)
library(ggplot2)

# Settings ----------------------------------------------------------------

# run options

options.traintype = 1 # Feature or Space

# task options
options.task = 1 # 1 = Motion Detection, 2 = Visual Search
options.TestTrain = 1 # 1 = test, 2 = train

# analysis options
n_days = switch (options.TestTrain, 2, 3)

paletteuse = "BrBG" #"PRGn"

# strings
strings.day = c("day-1", "day-4")
strings.task = c("AttnNFMotion",  "AttnNFVisualSearch");
strings.TestTrain = c("Test", "Train");
strings.traintype = c("Feature", "Space")

# Directories -------------------------------------------------------------

# set root directories
direct.Root = "Z:/" # VISATTNNF-Q1357/
direct.resultsRoot = paste(direct.Root, "Results/", sep = "")
direct.results = paste(direct.resultsRoot, "Train", strings.traintype[options.traintype], "/ALL/", sep = "")

# set filenames

filename.bids.results = paste( "SelectivityResults", sep = "")


# Get Data ----------------------------------------------------------------

file.ls = list.files(path=direct.results, pattern=glob2rx(paste( filename.bids.results, "*.mat", sep = "")))

DATA_ALL = readMat(paste(direct.results, tail(file.ls, n=1), sep = ""))

# assign dat
SELECTIVITY = DATA_ALL$Selectivity.ALL

## Preallocate bayes factors
n.subs = length(SELECTIVITY[1,1,])
BAYESPLOT = data.frame()
cuetype = 1

## Compute bayes factors

for (SS in 2:n.subs) {
  bf = ttestBF(SELECTIVITY[cuetype,1,1:SS], SELECTIVITY[cuetype,2,1:SS], paired = TRUE)
  dat=data.frame(bf, stringsAsFactors = FASLE)
  
  BAYESPLOT = rbind(BAYESPLOT, data.frame(BF = dat$bf))
  
}

## plot

Subjects =  2:n.subs

i <- ggplot(BAYESPLOT, aes( x=Subjects, y=BF)) 
i <- i + geom_line() +geom_point() 
tit = paste("Train ", strings.traintype[options.traintype], " Bayes Factor by Subject Number")
i <- i + ggtitle( tit ) + theme(plot.title = element_text(hjust = 0.5))

plot(i)
ggsave(paste(direct.results, "/", tit, ".png", sep=""))

