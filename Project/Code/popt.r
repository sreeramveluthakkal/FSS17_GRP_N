library(FFTrees)
library(randomForest)
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

trainRF<- function(test, train){
  model <- randomForest(formula = factor(bug) ~ wmc+dit+noc+cbo+rfc+lcom+ca+ce+npm+lcom3+loc+moa+mfa+cam+ic+cbm+amc+max_cc+avg_cc,
                        data = train,                
                        test = test,
                        ntree=1000,
                        nodesize=25,
                        main = "Bug Detector", 
                        decision.labels = c("No Bug", "Bug"),
                        importance=TRUE)
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
    if((predicted[i]==TRUE||predicted[i]==1)&& actual[i,"bug"]==1){
      tp <- tp + 1
    }
    if((predicted[i]==FALSE||predicted[i]==0)&& actual[i,"bug"]==1){
      fn <- fn + 1
    }
    if((predicted[i]==TRUE||predicted[i]==1)&& actual[i,"bug"]==0){
      fp <- fp + 1
    }
    if((predicted[i]==TRUE||predicted[i]==1)&& actual[i,"bug"]==0){
      fp <- fp + 1
    }
  }
  
  prec <- tp/(tp+fp)
  rec<- tp/(tp+fn)
  acc<- tp/num
  #cat("prec: ",prec, "recall: ",rec)
  output <- list("precision" = prec, "recall" = rec,"accuracy" = acc)
}



calculatePOpt<- function(dataSet, model_fft, model_rf){
  
  prediction_result_fft <- predict(model_fft, data = dataSet)
  prediction_result_rf <- predict(model_rf, data = dataSet)
  
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
  
  prediction_vector_fft <- data.frame("loc" = integer(), "bug" = integer() )
  for (i in 1: nrow(dataSet)){
    prediction_vector_fft[i,] <- c(dataSet[i,"loc"],if (prediction_result_fft[i]==TRUE)1 else 0 )
  }
  
  prediction_vector_rf <- data.frame("loc" = integer(), "bug" = integer() )
  for (i in 1: nrow(dataSet)){
    prediction_vector_rf[i,] <- c(dataSet[i,"loc"],if (prediction_result_rf[i]==1)1 else 0 )
  }
  
  
  sorted_prediction_vector_fft <- prediction_vector_fft[with(prediction_vector_fft,order(-bug,loc)),]
  sorted_prediction_vector_rf <- prediction_vector_rf[with(prediction_vector_rf,order(-bug,loc)),]
  
  predict_x_points_fft <- cumsum(sorted_prediction_vector_fft$loc) # x: LOC%
  predict_Xs_fft <- predict_x_points_fft/predict_x_points_fft[nrow(sorted_prediction_vector_fft)]
  predict_y_points_fft <- cumsum(sorted_prediction_vector_fft$bug) # x: LOC%
  predict_Ys_fft <- predict_y_points_fft/predict_y_points_fft[nrow(sorted_prediction_vector_fft)]
  
  predict_x_points_rf <- cumsum(sorted_prediction_vector_rf$loc) # x: LOC%
  predict_Xs_rf <- predict_x_points_rf/predict_x_points_rf[nrow(sorted_prediction_vector_rf)]
  predict_y_points_rf <- cumsum(sorted_prediction_vector_rf$bug) # x: LOC%
  predict_Ys_rf <- predict_y_points_rf/predict_y_points_rf[nrow(sorted_prediction_vector_rf)]
  
  pos_20pc_predict_fft <- min(which(predict_Xs_fft >= 0.2))
  cat("FFT predict 20 pc at ",pos_20pc_predict_fft)
  predict_x_points_fft[pos_20pc_predict_fft]
  
  pos_20pc_predict_rf <- min(which(predict_Xs_rf >= 0.2))
  cat("RF predict 20 pc at ",pos_20pc_predict_rf)
  predict_x_points_rf[pos_20pc_predict_rf]
  
  stats_fft <- getStats(prediction_result_fft,dataSet,pos_20pc_predict_fft)
  stats_rf <- getStats(prediction_result_rf,dataSet,pos_20pc_predict_rf)
  
  p_fft<- stats_fft$precision
  r_fft<-stats_fft$recall
  a_fft<- stats_fft$accuracy
  
  p_rf<- stats_rf$precision
  r_rf<-stats_rf$recall
  a_rf<- stats_rf$accuracy
  
  plot(optimal_Xs,optimal_Ys,type="l",xlab = "% code churn", ylab = "% defects detected")
  lines(worst_Xs,worst_Ys,type="l")
  lines(predict_Xs_fft,predict_Ys_fft,type="l", col = "blue")
  lines(predict_Xs_rf,predict_Ys_rf,type="l", col = "red")
  abline(0,1)
  abline(v=.2)
  
  area_under_optimal_curve_20pc <- get20pcAUC(optimal_Xs, optimal_Ys, pos_20pc_optimal)
  area_under_worst_curve_20pc <- get20pcAUC(worst_Xs, worst_Ys, pos_20pc_worst)
  
  area_under_prediction_curve_fft_20pc <- get20pcAUC(predict_Xs_fft, predict_Ys_fft, pos_20pc_predict_fft)
  area_under_prediction_curve_rf_20pc <- get20pcAUC(predict_Xs_rf, predict_Ys_rf, pos_20pc_predict_rf)
  
  pOpt_fft <- getPOPTNormalized(area_under_optimal_curve_20pc,area_under_prediction_curve_fft_20pc,area_under_worst_curve_20pc)
  pOpt_rf <- getPOPTNormalized(area_under_optimal_curve_20pc,area_under_prediction_curve_rf_20pc,area_under_worst_curve_20pc)
  
  output<- list("popt_fft" = pOpt_fft,"precision_fft" = p_fft, "recall_fft" = r_fft,"accuracy_fft"= a_fft,
                "popt_rf" = pOpt_rf, "precision_rf" = p_rf, "recall_rf" = r_rf,"accuracy_rf"= a_rf)
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
  pOpt_vector_fft<-c()
  precision_vector_fft<-c()
  recall_vector_fft<-c()
  acc_vector_fft<-c()
  
  pOpt_vector_rf<-c()
  precision_vector_rf<-c()
  recall_vector_rf<-c()
  acc_vector_rf<-c()
  for(input_size in data_size_vector ){
    cat("looping: ",input_size)
    data<- getData("../Data/velocity_m.csv",input_size,seed)
    
    train <- data$trainData
    test <-data$testData
    model_fft<- trainFFT(train,test)
    model_rf<- trainRF(train,test)
    
    pOpt<- calculatePOpt(test,model_fft,model_rf)
    
    pOpt_vector_rf<-c(pOpt_vector_rf,pOpt$popt_rf)
    precision_vector_rf<-c(precision_vector_rf,pOpt$precision_rf)
    recall_vector_rf<-c(recall_vector_rf,pOpt$recall_rf)
    acc_vector_rf<-c(acc_vector_rf,pOpt$accuracy_rf)
    
    pOpt_vector_fft<-c(pOpt_vector_fft,pOpt$popt_fft)
    precision_vector_fft<-c(precision_vector_fft,pOpt$precision_fft)
    recall_vector_fft<-c(recall_vector_fft,pOpt$recall_fft)
    acc_vector_fft<-c(acc_vector_fft,pOpt$accuracy_fft)
  }
  
  plot(data_size_vector, pOpt_vector_fft,type = "l",xlab = "no of records", ylab = "pOpt",col = 'blue',ylim = c(0,1))
  lines(data_size_vector, pOpt_vector_rf,type = "l",col='red',ylim = c(0,1))
  
  plot(data_size_vector, precision_vector_fft,type = "l",xlab = "no of records", ylab = "PRECISION",col = 'blue',ylim = c(0,1))
  lines(data_size_vector, precision_vector_rf,type = "l",col='red',ylim = c(0,1))
  
  plot(data_size_vector, recall_vector_fft,type = "l",xlab = "no of records", ylab = "RECALL",col = 'blue',ylim = c(0,1))
  lines(data_size_vector, recall_vector_rf,type = "l",col='red',ylim = c(0,1))
  
  print("here...")
  print(acc_vector_fft)
  print(acc_vector_rf)
  plot(data_size_vector, acc_vector_fft,type = "l",xlab = "no of records", ylab = "ACC",col = 'blue',ylim = c(0,1))
  lines(data_size_vector, acc_vector_rf,type = "l",col='red',ylim = c(0,1))
  
  
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

f1_main<- (2*precissionMain*recallMain)/(precissionMain + recallMain)

#plot(data_size_vector,pOptMain,type = "l",xlab = " # of records ", ylab = "pOpt")
#plot(data_size_vector,recallMain,type = "l",xlab = " # of records ", ylab = "recall")
#plot(data_size_vector,precissionMain,type = "l",xlab = " # of records ", ylab = "precission")
#plot(data_size_vector,accuracyMain,type = "l",xlab = " # of records ", ylab = "accuracy")
#plot(data_size_vector,f1_main,type = "l",xlab = " # of records ", ylab = "F1")


