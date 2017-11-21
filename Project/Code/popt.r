library(FFTrees)
library(zoo)

getData<- function(filename, rows,seed_val){
  bugs<- read.csv(file=filename, header=TRUE, sep=",", nrows=rows)
  #Remove unnecessary columns
  set.seed(seed_val)
  bugs<- bugs[-c(1,2,3)]
  #bugs <- bugs[sample(nrow(bugs)),]
  # Randomize test set to test/train sections
  # https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
  smp_size <- floor(0.8 * nrow(bugs))
  ## set the seed to make your partition reproductible
  set.seed(123)
  train_ind <- sample(seq_len(nrow(bugs)), size = smp_size)
  train <- bugs[train_ind, ]
  test <- bugs[-train_ind, ]
  #train <- bugs[1:smp_size, ]
  #test <- bugs[smp_size+1:rows, ]
  output <- list("trainData" = train, "testData" = test)
}

trainFFT<- function(test, train){
  model <- FFTrees(formula = bug ~ wmc+dit+noc+cbo+rfc+lcom+ca+ce+npm+lcom3+loc+moa+mfa+cam+ic+cbm+amc+max_cc+avg_cc,
                   data = train,                
                   data.test = test,        
                   main = "Bug Detector",
                   do.comp = FALSE,
                   decision.labels = c("No Bug", "Bug"))
}


getAUC<- function(x_axis,y_axis){
  id <- order(x_axis)
  sum(diff(x_axis[id])*rollmean(y_axis[id],2))
}

#https://stackoverflow.com/questions/4954507/calculate-the-area-under-a-curve
get20pcAUC<- function(x_axis,y_axis,cutoff){
  x_axis <- x_axis[1:cutoff]
  y_axis <- y_axis[1:cutoff]
  id <- order(x_axis)
  sum(diff(x_axis[id])*rollmean(y_axis[id],2))
}

getPOPTNormalized <- function(area_optimal, area_model, area_worst){
  1 - ( (area_optimal - area_model)/(area_optimal - area_worst))
}


getStats<- function(predicted, actual, num){
  
  tp<-0
  fn<-0
  fp<-0
  
  for(i in 1:num){
    if(predicted[i]==TRUE&& actual[i,"bug"]==1){
      tp <- tp + 1
    }
    if(predicted[i]==FALSE&& actual[i,"bug"]==1){
      fn <- fn + 1
    }
    if(predicted[i]==TRUE&& actual[i,"bug"]==0){
      fp <- fp + 1
    }
    if(predicted[i]==TRUE&& actual[i,"bug"]==0){
      fp <- fp + 1
    }
  }
  
  prec <- tp/(tp+fp)
  rec<- tp/(tp+fn)
  acc<- tp/num
  #cat("prec: ",prec, "recall: ",rec)
  output <- list("precision" = prec, "recall" = rec,"accuracy" = acc)
}



calculatePOpt<- function(dataSet, model){
  
  prediction_result <- predict(model, data = dataSet)
  
  
  df_optimal <- dataSet[with(dataSet,order(-bug,loc)),]
  opt_x_points <- cumsum(df_optimal$loc) # x: LOC%
  optimal_Xs <- opt_x_points/opt_x_points[nrow(df_optimal)]
  opt_y_points <- cumsum(df_optimal$bug) # x: LOC%
  optimal_Ys <- opt_y_points/opt_y_points[nrow(df_optimal)]
  
  pos_20pc_optimal <- min(which(optimal_Xs >= 0.2))
  opt_x_points[pos_20pc_optimal]
  
  
  df_worst <- dataSet[with(dataSet,order(bug,-loc)),]
  wst_x_points <- cumsum(df_worst$loc) # x: LOC%
  worst_Xs <- wst_x_points/wst_x_points[nrow(df_worst)]
  wst_y_points <- cumsum(df_worst$bug) # x: LOC%
  worst_Ys <- wst_y_points/wst_y_points[nrow(df_worst)]
  
  
  pos_20pc_worst <- min(which(worst_Xs >= 0.2))
  pos_20pc_worst
  wst_x_points[pos_20pc_worst]
  
  prediction_vector <- data.frame("loc" = integer(), "bug" = integer() )
  for (i in 1: nrow(dataSet)){
    prediction_vector[i,] <- c(dataSet[i,"loc"],if (prediction_result[i]==TRUE)1 else 0 )
  }
  
  
  sorted_prediction_vector <- prediction_vector[with(prediction_vector,order(-bug,loc)),]
  
  predict_x_points <- cumsum(sorted_prediction_vector$loc) # x: LOC%
  predict_Xs <- predict_x_points/predict_x_points[nrow(sorted_prediction_vector)]
  predict_y_points <- cumsum(sorted_prediction_vector$bug) # x: LOC%
  predict_Ys <- predict_y_points/predict_y_points[nrow(sorted_prediction_vector)]
  
  pos_20pc_predict <- min(which(predict_Xs >= 0.2))
  cat("predict 20 pc at ",pos_20pc_predict)
  predict_x_points[pos_20pc_predict]
  
  stats<- getStats(prediction_result,dataSet,pos_20pc_predict)
  p<- stats$precision
  r<-stats$recall
  a<- stats$accuracy
  
  plot(optimal_Xs,optimal_Ys,type="l",xlab = "% code churn", ylab = "% defects detected")
  lines(worst_Xs,worst_Ys,type="l")
  lines(predict_Xs,predict_Ys,type="l", col = "red")
  abline(0,1)
  abline(v=.2)
  
  area_under_optimal_curve_20pc <- get20pcAUC(optimal_Xs, optimal_Ys, pos_20pc_optimal)
  area_under_prediction_curve_20pc <- get20pcAUC(predict_Xs, predict_Ys, pos_20pc_predict)
  area_under_worst_curve_20pc <- get20pcAUC(worst_Xs, worst_Ys, pos_20pc_worst)
  
  pOpt <- getPOPTNormalized(area_under_optimal_curve_20pc,area_under_prediction_curve_20pc,area_under_worst_curve_20pc)
  output<- list("popt" = pOpt, "precision" = p, "recall" = r,"accuracy"= a)
}

#Load Data

data_size_vector<- c(1000,2000,4000,6000,8000)

popt_sum<-0
recall_sum<-0
precission_sum<-0
accuracy_sum<-0
seed_vector<-c(123)
times<- length(seed_vector)

pOptMain<-rep(0,times)
recallMain<-rep(0,times)
accuracyMain<-rep(0,times)
precissionMain<-rep(0,times)

for(seed in seed_vector ){
  pOpt_vector<-c()
  precision_vector<-c()
  recall_vector<-c()
  acc_vector<-c()
  
  for(input_size in data_size_vector ){
    data<- getData("../Data/velocity_m.csv",input_size,seed)
    
    train <- data$trainData
    test <-data$testData
    model<- trainFFT(train,test)
    pOpt<- calculatePOpt(test,model)
    pOpt_vector<-c(pOpt_vector,pOpt$popt)
    precision_vector<-c(precision_vector,pOpt$precision)
    recall_vector<-c(recall_vector,pOpt$recall)
    acc_vector<-c(acc_vector,pOpt$accuracy)
  }
  
  cat("pOpt vector is ",pOpt_vector)
  cat("recall vector is ",recall_vector)
  cat("precision vector is ",precision_vector)
  cat("accuracy vector is ",acc_vector)
  
  plot(data_size_vector,pOpt_vector,type = "l",xlab = "no of records", ylab = "pOpt")
  #plot(data_size_vector,precision_vector,type = "l",xlab = "no of records", ylab = "PRECISION")
  #plot(data_size_vector,recall_vector,type = "l",xlab = "no of records", ylab = "RECALL")
  #plot(data_size_vector,acc_vector,type = "l",xlab = "no of records", ylab = "ACCURACY")
  
  pOptMain<- pOptMain + pOpt_vector
  precissionMain<- precissionMain + precision_vector
  accuracyMain<- accuracyMain + acc_vector
  recallMain<- recallMain + recall_vector
}
cat("pOpt main is ",pOptMain)
pOptMain <- pOptMain/times
recallMain <- recallMain/times
precissionMain <- precissionMain/times
accuracyMain <- accuracyMain/times
cat("pOpt mean is ",pOptMain)

f1_main<- (2*precissionMain*recallMain)/(precissionMain + recallMain)

plot(data_size_vector,pOptMain,type = "l",xlab = " # of records ", ylab = "pOpt")
plot(data_size_vector,recallMain,type = "l",xlab = " # of records ", ylab = "recall")
plot(data_size_vector,precissionMain,type = "l",xlab = " # of records ", ylab = "precission")
plot(data_size_vector,accuracyMain,type = "l",xlab = " # of records ", ylab = "accuracy")
plot(data_size_vector,f1_main,type = "l",xlab = " # of records ", ylab = "F1")


