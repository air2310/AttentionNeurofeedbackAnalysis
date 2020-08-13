library(ggplot2)
library(reshape2)
library(plyr)
library(RColorBrewer)
library(ggpubr)
library(ggthemes)

## Subnumbers
options.traintype = 1 # Feature or Space

if (options.traintype == 1) {
  subnumbers = c(1, 2 , 4, 8, 9, 18, 21, 23, 41, 47 )
} else {
  subnumbers = c(10, 11, 19, 22, 28, 29, 43, 45, 46, 49)
}

# Settings ----------------------------------------------------------------

# run options
options.TestTrain = 1 # 1 = test, 2 = train

# task options
options.task = 1 # 1 = Motion Detection

# analysis options
n_days = switch (options.TestTrain, 2, 3)

paletteuse = "Dark2"

yellow = rgb(242 / 255, 176 / 255, 53 / 255)
orange = rgb(236 / 255, 85 / 255, 58 / 255)
darkteal = rgb(18 / 255, 47 / 255, 65 / 255)
medteal = rgb(5 / 255, 133 / 255, 134 / 255)
lightteal = rgb(78 / 255, 185 / 255, 159 / 255)
colsuse2 = c(yellow, orange)


# strings
strings.day = c("pre-training", "post-training")
strings.task = c("AttnNFMotion");
strings.TestTrain = c("Test", "Train");
strings.traintype = c("Feature", "Space")
strings.behave = c("miss", "incorrect", "falsealarm", "correct")

# response options
responseopts.miss = 0
responseopts.incorrect = 1
responseopts.falsealarm = 2
responseopts.correct = 3

# Directories -------------------------------------------------------------

# set root directories
direct.Root = "//data.qbi.uq.edu.au/VISATTNNF-Q1357/"
direct.resultsRoot = paste(direct.Root, "Results/", sep = "")
direct.resultsGroup = paste(direct.Root, "Results/group", sep = "")
direct.resultsGroup = paste(direct.resultsRoot, "Train",strings.traintype[options.traintype], "/group",sep = "")
ACC_Group = data.frame()
RT_Group = data.frame()

for (SUB in 1:length(subnumbers)) {
  subject = subnumbers[SUB]
  
  # get subject string
  if (subject < 10) {
    bids.substring = paste('sub-0',  toString(subject), sep = "")
  } else {
    bids.substring = paste('sub-',  toString(subject), sep = "")
  }
  
  # get directories
  direct.results = paste(direct.resultsRoot, "Train",strings.traintype[options.traintype], "/",bids.substring,sep = "")
  
  # get data
  savefilename = paste(direct.results, "/", bids.substring, "Behave_pre_post_results.rds", sep = "")
  RESULTS=readRDS(file = savefilename)
  
  # accumulate accuracy across subjects
  acctmp = RESULTS['ACC']
  acctmp = cbind(acctmp, data.frame(sub=SUB))
  ACC_Group = rbind(ACC_Group, acctmp)
  
  rttmp = RESULTS['RT']
  rttmp = cbind(rttmp, data.frame(sub=SUB))
  RT_Group = rbind(RT_Group, rttmp)
}

# plot group accuracy bars -------------------------------------

colsuse <- brewer.pal(4, 'BrBG')
# ACC_Group$ACC.attntype<- factor(ACC_Group$ACC.attntype, levels = c(1, 2), labels = c("Feature", "Space"))

# plot
i <- ggplot(ACC_Group, aes(x=ACC.days, y=ACC.percent, fill = as.factor(ACC.behave))) 
i <- i + geom_bar(stat='identity', width = 0.7, alpha = 0.5)  
i <- i + facet_wrap(~ACC.attntype)
i <- i + scale_fill_manual(values =colsuse , labels = strings.behave, name = "response")
i <- i + scale_y_continuous(expand = c(0, 0))
i <- i + ylab("Percentage (%)")
i <- i + ggtitle( paste("Group SDT data by training day - Train: ", strings.traintype[options.traintype], setp="")) + theme(plot.title = element_text(hjust = 0.5))

plot(i)

ggsave(paste(direct.resultsGroup, "/", "Group SDT by training day and attentype.png", sep=""))


# Function to calculate the mean and std of each group -------------------------------------

data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE)/length(x[[col]]))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  return(data_sum)
}

datplot = subset(ACC_Group, ACC.behave==3)
datstats <- data_summary(datplot, varname="ACC.percent", 
                    groupnames=c("ACC.days", "ACC.attntype"))

datstats$ACC.attntype <- factor(datstats$ACC.attntype,levels = c("Space", "Feature")) # reorder space and feature
datstats$ACC.days <- factor(datstats$ACC.days, levels = c("day-1", "day-4"), labels = strings.day) # rename days



# plot bar graph of just accuracy

j<-  ggplot(datstats, aes(x=ACC.attntype, y=mean, fill = ACC.days)) 
j <- j + geom_bar(stat='identity',alpha = 1, position=position_dodge())  
j <- j + geom_errorbar(aes(ymin=mean-sd, ymax=mean+sd), width=.2, position=position_dodge(.9))
# j <- j + geom_dotplot(datplot, aes(x=ACC.attntype, y=ACC,percent, alpha=0.5))
j <- j + ylab("Percentage (%)")
j <- j + scale_fill_manual(values = colsuse2 ) + theme_classic()
j <- j + scale_y_continuous(expand = c(0, 0))
j <- j + ggtitle( paste("Group Accuracy data by training day - Train: ", strings.traintype[options.traintype], setp="")) + theme(plot.title = element_text(hjust = 0.5))
j <- j + ylim(0,60)
plot(j)

ggsave(paste(direct.resultsGroup, "/", "Group Accuracy by training day and attentype.png", sep=""))



## Plot Average Reaction Time Data -------------

data_summary <- function(data, varname, groupnames){
  require(plyr)
  summary_func <- function(x, col){
    c(mean = mean(x[[col]], na.rm=TRUE),
      sd = sd(x[[col]], na.rm=TRUE)/length(x[[col]]))
  }
  data_sum<-ddply(data, groupnames, .fun=summary_func,
                  varname)
  return(data_sum)
}

datstats <- data_summary(RT_Group, varname="RT.RT", 
                         groupnames=c("RT.days", "RT.attntype", "sub"))

datstats$RT.attntype <- factor(datstats$RT.attntype, levels = c(2, 1), labels = c("Space", "Feature")) # reorder space and feature
datstats$RT.days <- factor(datstats$RT.days, levels = c("day-1", "day-4"), labels = strings.day) # rename days


h <- ggplot(datstats, aes(x=RT.attntype, y=mean, color = RT.days, fill = RT.days, stat(count)))  + geom_violin(alpha = 0.7, position=position_dodge(1)) 
# h <- h + geom_dotplot(binaxis='y', stackdir='center', position=position_dodge(1))

h <- h + stat_summary(fun.y=mean, geom="point", shape=23, size=2, position=position_dodge(1))
h <- h +scale_colour_manual(values = colsuse2 ) + scale_fill_manual(values = colsuse2 )

h <- h + ylab("Reaction Time (s)")
# h <- h + ggtitle("Reaction time by training day") + theme(plot.title = element_text(hjust = 0.5))
h <- h + ggtitle( paste("Reaction time by training day- Train: ", strings.traintype[options.traintype], setp="")) + theme(plot.title = element_text(hjust = 0.5))
#h <- h + theme_tufte(base_size = 20) + theme(aspect.ratio=1, legend.position = "none", plot.margin = unit( c(0,0,0,0), "cm"))
h <- h + scale_y_continuous(expand = c(0, 0))
# h <- h + ylim(0,2)


plot(h)

ggsave(paste(direct.resultsGroup, "/", "RT by training day violin.png", sep=""))


# To Do:
# figure out axis - why out of 800 not 100%?
# plot correct percentage with errorbars - as violin plot?
# get averaged reaction time data. 
