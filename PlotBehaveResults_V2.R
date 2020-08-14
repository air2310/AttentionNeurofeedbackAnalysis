library(R.matlab)
library(ggplot2)
library(reshape2)
library(plyr)
library(RColorBrewer)
library(ggpubr)
library(ggthemes)
library(tidyverse)

## Subnumbers
options.traintype =2 # Feature or Space

if (options.traintype == 1) {
  subnumbers = c(1, 2 , 4, 8, 9, 18, 21, 23, 41, 47 )
} else {
  subnumbers = c(10, 11, 19, 22, 28, 29, 43, 45, 46, 49, 52, 53)
}
# Settings ----------------------------------------------------------------

for (SUB in length(subnumbers)) { #6:length(subnumbers
  subject = subnumbers[SUB]
  options.TestDay = 1;
  
  # run options
  options.TestTrain = 1 # 1 = test, 2 = train
  
  # task options
  options.task = 1 # 1 = Motion Detection, 2 = Visual Search
  
  # analysis options
  n_days = switch (options.TestTrain, 2, 3)
  
  # paletteuse = "BrBG" #"PRGn"
  paletteuse = "Dark2"
  
  # strings
  strings.day = c("day-1", "day-4")
  strings.task = c("AttnNFMotion",  "AttnNFVisualSearch");
  strings.TestTrain = c("Test", "Train");
  strings.traintype = c("Feature", "Space")
  strings.behave = c("miss", "incorrect", "falsealarm", "correct")
  
  # response options
  responseopts.miss = 0
  responseopts.incorrect = 1
  responseopts.falsealarm = 2
  responseopts.correct = 3
  
  
  # preallocate behavioural variables
  
  ACCURACY_ALL = data.frame()
  RT_ALL = data.frame()
  daystr_ALL = data.frame()
  ATTNTYPE_ALL= data.frame()
  
  testdays = c(1,4)
  
  for(options.TestDayloop in 1:n_days){
      options.TestDay = testdays[options.TestDayloop]
        
      # Directories -------------------------------------------------------------
      
      # set root directories
      direct.Root = "//data.qbi.uq.edu.au/VISATTNNF-Q1357/"
      direct.dataRoot = paste(direct.Root, "Data/", sep = "")
      direct.resultsRoot = paste(direct.Root, "Results/", sep = "")
      
      # get subject string
      if (subject < 10) {
        bids.substring = paste('sub-0',  toString(subject), sep = "")
      } else {
        bids.substring = paste('sub-',  toString(subject), sep = "")
      }
      
      # get case strings
      bids.taskname = strings.task[options.task] # Unique name of task for saving
      bids.casestring = paste(bids.substring, "_task-", bids.taskname, "_", strings.day[options.TestDayloop], "_phase-", strings.TestTrain[options.TestTrain], sep = "")
      
      # get directories
      
      direct.data_bids_behave =  paste(direct.dataRoot, "Train", strings.traintype[options.traintype], "/", bids.substring, "/behave/", sep = "")
      direct.results = paste(direct.resultsRoot, "Train", strings.traintype[options.traintype], "/", bids.substring, sep = "")
      
      # set filenames
      
      filename.bids.BEHAV = paste(bids.casestring, "_behav", sep = "")
      
      
      # Get Data ----------------------------------------------------------------
      
      file.ls <- list.files(path=direct.data_bids_behave, pattern=glob2rx(paste("R", filename.bids.BEHAV, "*.mat", sep = "")))
      
      DATA_ALL = readMat(paste(direct.data_bids_behave, tail(file.ls, n=1), sep = ""))
  
      
      # Extract Variables -------------------------------------------------------
      
      n.trials = as.numeric(DATA_ALL$n[2])
      mon.ref = as.numeric(DATA_ALL$mon[2])
      
      responseperiod = c(0.3, 1.5)*mon.ref
      directions = c(0, 90, 180, 270)
      
      RESPONSETIME = data.frame(DATA_ALL$RESPONSETIME)
      
      # Gather Responses --------------------------------------------------------
      
      DATA = DATA_ALL$DATA
      Dirchanges = data.frame(DATA[12])
      MoveOnsets = data.frame(DATA[11])
      trialattntype = data.frame(DATA[5])
      
      RESPONSE = data.frame(DATA_ALL$RESPONSE)
      dat = rbind(rep(0, n.trials), data.frame(diff(as.matrix(RESPONSE))))
      
      response = data.frame()
      for (TT in 1:n.trials) {
        idx = which(dat[TT]>0)
        
        if(length(idx)>0) {
          for (ii in 1:length(idx)) {
            # stack response, trial, and indext of all responses together
            tmp = cbind(data.frame(response = RESPONSE[idx[ii],TT]), TT, data.frame(index = idx[ii]))
            response = rbind(response, tmp)
          }
        }
      }
      
      Moveorder = data.frame(DATA[6])
      
      
      # Score Behaviour ---------------------------------------------------------
      
      ACCURACY = data.frame()
      ATTNTYPE = data.frame()
      RT = data.frame()
      daystr = data.frame()
      
      if(length(response)>0) {
        for(TT in 1:n.trials){
          
          # Score correct answer
          if (trialattntype[TT,] == 1) { # feature
             correctmoveorder = which(Moveorder[TT,] == 1 | Moveorder[TT,] == 3 )
          }
          
          if (trialattntype[TT,] == 2) {  # space
             correctmoveorder = which(Moveorder[TT,] == 1 | Moveorder[TT,] == 2 )
          }
          
          
          correct = Dirchanges[TT,correctmoveorder]
          for(ii in 1:length(correct)){
            correct[ii]=which(is.element(directions, correct[ii]))
          }
          
          # define answer period
          tmp = MoveOnsets[TT,correctmoveorder]
          moveframes = data.frame()
          for(ii in 1:length(tmp)){
            tmp2 = data.frame( t(c(tmp[ii] + responseperiod[1],tmp[ii] + responseperiod[2] )))
            names(tmp2) = c("start", "stop")
            moveframes = rbind(moveframes, tmp2)
          }
          
          #  gather Responses from this trial
          trialresponses = which(response[,2]==TT)
          trialresponses_accounted = rep(0, length(trialresponses))
          
          # get responses during target response period
          for(ii in 1:length(correctmoveorder)) {
            
            # get eligible responses - within trial response periods
            idx = which(response[,2] == TT & response[,3] > moveframes[ii,1] & response[,3] < moveframes[ii,2])
            
            if(length(idx)>0) { # if any identified - mark accounted for
              trialresponses_accounted[is.element(trialresponses, idx)] = 1
            }
            
            #  fill accuracy data
            if(length(idx)==0){
              ACCURACY = rbind(ACCURACY, data.frame(ACC= responseopts.miss))
              RT = rbind(RT, data.frame(RT=NaN))
              ATTNTYPE = rbind(ATTNTYPE, data.frame(attntype= trialattntype[TT,]))
              
            } else if (length(idx) > 1) { # false alarm somewhere
              # Accuracy
              tmp = rep(NaN, length(idx))
              tmp[is.element(response[idx,1], correct[ii])] = responseopts.correct
              tmp[!is.element(response[idx,1], correct[ii])] = responseopts.falsealarm
              
              ACCURACY = rbind(ACCURACY, data.frame(ACC=tmp))
              
              # RT
              tmp = rep(NaN, length(idx))
              tmp[is.element(response[idx,1], correct[ii])] = 
              RESPONSETIME[response[idx[is.element(response[idx,1], correct[ii])],3],TT] - MoveOnsets[TT,correctmoveorder[ii]]/mon.ref
              # RESPONSETIME(response(idx(ismember(tmpacc , responseopts.correct)),3),TT) -  DATA.MOVEONSETS(TT,correctmoveorder(ii))/mon.ref;
              
              RT = rbind(RT, data.frame(RT = tmp))
              ATTNTYPE = rbind(ATTNTYPE, data.frame(attntype = rep(trialattntype[TT,], length(idx))))
              
              
              #Rt stuff here
            } else if (response[idx,1] == correct[ii]) {
              ACCURACY = rbind(ACCURACY, data.frame(ACC=responseopts.correct))
              RT = rbind(RT, data.frame(RT=RESPONSETIME[response[idx,3],TT] - MoveOnsets[TT,correctmoveorder[ii]]/mon.ref))
              ATTNTYPE = rbind(ATTNTYPE, data.frame(attntype= trialattntype[TT,]))
            
              
              
            } else {
              ACCURACY = rbind(ACCURACY, data.frame(ACC=responseopts.incorrect))
              RT = rbind(RT, data.frame(RT=RESPONSETIME[response[idx,3],TT] - MoveOnsets[TT,correctmoveorder[ii]]/mon.ref))
              ATTNTYPE = rbind(ATTNTYPE, data.frame(attntype= trialattntype[TT,]))
            }                   
          }
          
        if(any(trialresponses_accounted == 0)) {
          # deal with these here
          numFAs = sum(trialresponses_accounted==0)
          ACCURACY = rbind(ACCURACY, data.frame(ACC=rep(responseopts.falsealarm, numFAs)))
          RT = rbind(RT, data.frame(RT = rep(NaN, numFAs)))
          ATTNTYPE = rbind(ATTNTYPE, data.frame(attntype= rep(trialattntype[TT,], numFAs)))
        }
          
        }
      }
      
      # collate together across days
      
      n.responses = lengths(ACCURACY)
      daystr_ALL = rbind(daystr_ALL, data.frame(days = rep(strings.day[options.TestDayloop], n.responses )))
      ACCURACY_ALL = rbind(ACCURACY_ALL, ACCURACY)
      RT_ALL = rbind(RT_ALL, RT)
      ATTNTYPE_ALL = rbind(ATTNTYPE_ALL, ATTNTYPE)
  
  }
  
  # get rid of impossible RTs ---------------------------------
  
  ACCURACY_ALL  [RT_ALL>2.0] = NaN
  ATTNTYPE_ALL  [RT_ALL>2.0] = NaN
  daystr_ALL [RT_ALL>2.0 ] = NaN
  RT_ALL[RT_ALL>2.0] = NaN
  
  # bind together reaction time plotting data
  
  tmp = RT_ALL
  tmp[ACCURACY_ALL==responseopts.incorrect  ] = NaN
  tmp[ACCURACY_ALL==responseopts.falsealarm] = NaN
  RTplot = cbind(tmp, daystr_ALL, ATTNTYPE_ALL)
  RTplot = na.omit(RTplot)
  
  
  ACCplot = cbind(ACCURACY_ALL, daystr_ALL, ATTNTYPE_ALL)
  ACCplot = na.omit(ACCplot)
  
  # Get reaction time Means
  mu <- ddply(RTplot, "days", summarise, grp.mean=mean(RT))
  
  
  # plot Reactiontime by day and attntype
  
  RTplot3 = RTplot
  RTplot3$attntype = as.factor(RTplot$attntype)
  RTplot3$days = as.factor(RTplot$days)
  
  paletteuse = "Dark2"
  
  h <- ggplot(RTplot3, aes(x=attntype, y= RT, color = days, fill = days, stat(count)))  + geom_violin(alpha = 0.3, position=position_dodge(1)) 
  
  # h <- h + geom_dotplot(binaxis='y', stackdir='center', position=position_dodge(1))
  
  h <- h + stat_summary(fun.y=mean, geom="point", shape=23, size=2, position=position_dodge(1))
  
  h <- h + scale_color_brewer(palette = paletteuse) + scale_fill_brewer(palette = paletteuse) 
  
  h <- h + ylab("Reaction Time (s)")
  h <- h + ggtitle("Reaction time by training day") + theme(plot.title = element_text(hjust = 0.5))
  #h <- h + theme_tufte(base_size = 20) + theme(aspect.ratio=1, legend.position = "none", plot.margin = unit( c(0,0,0,0), "cm"))
  h <- h + scale_y_continuous(expand = c(0, 0))
  h <- h + scale_x_discrete(labels=c("1" = "Feature", "2" = "Space"))
  h <- h + ylim(0,2)
  
  
  plot(h)
  # ggsave(paste(direct.results, "/", bids.substring, "_" , bids.taskname, "_Reaction_time_by_training_day", sep=""))
  ggsave(paste(direct.results, "/", "RT by training day violin2.png", sep=""))
  
  
  
  
  # plot accuracy stacked hist by attentiontype
  
  Acctmp= data.frame()
  daystmp = data.frame()
  behavetmp = data.frame()
  attntypetmp = data.frame()
  
  for (trialattntype in 1:2) {
    for(day in 1:2){
      for(behave in 0:3) {
        dayuse = strings.day[day]
        
        
        percentage = 100*sum(ACCplot$attntype == trialattntype & ACCplot$days==dayuse & ACCplot$ACC == behave)/sum(ACCplot$attntype == trialattntype &ACCplot$days==dayuse)
        
        Acctmp = rbind(Acctmp, data.frame(percent = percentage))
        daystmp = rbind(daystmp, data.frame(days = dayuse))
        behavetmp = rbind(behavetmp, data.frame(behave = behave))
        attntypetmp = rbind(attntypetmp, data.frame(attntype = trialattntype))
      }
    }
  }
  
  ACCPLOT2 = cbind(Acctmp, daystmp, behavetmp, attntypetmp )
  
  ACCPLOT2$attntype <- factor(ACCPLOT2$attntype, levels = c(1, 2), labels = c("Feature", "Space"))
  accsave = ACCPLOT2 # save out for later
  
  colsuse <- brewer.pal(4, 'BrBG')
  
  
  # plot
  i <- ggplot(ACCPLOT2, aes(x=days, y=percent, fill = as.factor(behave))) 
  i <- i + geom_bar(stat='identity', width = 0.7, alpha = 0.5)  
  i <- i + facet_wrap(~attntype)
  i <- i + scale_fill_manual(values =colsuse , labels = strings.behave, name = "response")
  i <- i + scale_y_continuous(expand = c(0, 0))
  i <- i + ylab("Percentage (%)")
  i <- i + ggtitle( "Accuracy data by training day") + theme(plot.title = element_text(hjust = 0.5))
  
  plot(i)
  ggsave(paste(direct.results, "/", "Accuracy by training day and attentype.png", sep=""))
  
  
  
  # Plot Reaction Time Data Violin ------------------------------------------
  
  RTplot$days = as.factor(RTplot$days)
  
  
  h <- ggplot(RTplot, aes(x=days, y = RT, color = days, fill = days, stat(count))) +   geom_violin(trim=FALSE, alpha = 0.3)
  h <- h + stat_summary(fun.y=mean, geom="point", shape=23, size=2)
  # h <- h + geom_boxplot(width=0.1)
  # h <- h + geom_jitter(shape=16, position=position_jitter(0.2))
  
  h <- h + scale_color_brewer(palette = paletteuse) + scale_fill_brewer(palette = paletteuse) 
  h <- h + theme_tufte(base_size = 20) + theme(aspect.ratio=1, legend.position = "none", plot.margin = unit( c(0,0,0,0), "cm"))
  h <- h + scale_y_continuous(expand = c(0, 0))
  
  h <- h + ylab("Reaction Time (s)")
  h <- h + ggtitle("Reaction time by training day") + theme(plot.title = element_text(hjust = 0.5))
  
  
  # Plot Accuracy Data stacked hist ------------------------------------------
  Acctmp= data.frame()
  daystmp = data.frame()
  behavetmp = data.frame()
  for(day in 1:2){
    for(behave in 0:3) {
      dayuse = strings.day[day]
      
      
      percentage = 100*sum(ACCplot$days==dayuse & ACCplot$ACC == behave)/sum(ACCplot$days==dayuse)
      
      Acctmp = rbind(Acctmp, data.frame(percent = percentage))
      daystmp = rbind(daystmp, data.frame(days = dayuse))
      behavetmp = rbind(behavetmp, data.frame(behave = behave))
    }
  }
  
  ACCPLOT2 = cbind(Acctmp, daystmp, behavetmp)
  colsuse <- brewer.pal(4, "BrBG")
  
  # plot
  i <- ggplot(ACCPLOT2, aes(x=days, y=percent, fill = as.factor(behave))) 
  i <- i + geom_bar(stat='identity', width = 0.7, alpha = 0.5)  
  
  i <- i + scale_fill_manual(values =colsuse , labels = strings.behave, name = "response")
  i <- i +   theme_tufte(base_size = 20) + theme(aspect.ratio=1, plot.margin = unit( c(0,0,0,0), "cm"))
  
  i <- i + scale_y_continuous(expand = c(0, 0))
  
  i <- i + ylab("Percentage (%)")
  i <- i + ggtitle( "Accuracy data by training day") + theme(plot.title = element_text(hjust = 0.5))
  
  
  # plot(i)
  # ggsave(paste(direct.results, "/", "ACCURACY by training day Stackhist.png", sep=""))
  
  # title <- expression(atop(bold(bids.substring), scriptstyle("This is the caption")))
  
  png(paste(direct.results, "/", "Acc and RT.png", sep=""), width = 30, height = 17, units = "cm", res = 300)
  j = egg::ggarrange(h, i,nrow = 1)
  # j = annotate_figure(j,top=text_grob(bids.substring, face = "bold", size = 20))
  dev.off()
  # 
  # plot(j)
  # ggsave(paste(direct.results, "/", "ACC & RT by training day combined.png", sep=""))
  
  
  
  # Save Results
  RESULTS = list(RT=RTplot, ACC =accsave)
  direct.results = paste(direct.resultsRoot, "Train", strings.traintype[options.traintype], "/", bids.substring, sep = "")
  
  savefilename = paste(direct.results, "/", bids.substring, "Behave_pre_post_results.rds", sep = "")
  saveRDS(RESULTS, file = savefilename)
}
