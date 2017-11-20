library(FFTrees)
library(zoo)
source('fft_common.r')

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

calculatePOpt<- function(dataSet, model){
  
  prediction <- predict(model, data = dataSet)
  
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
    prediction_vector[i,] <- c(dataSet[i,"loc"],if (prediction[i]==TRUE)1 else 0 )
  }
  #df[with(df,order(-bug,loc)),]
  sorted_prediction_vector <- prediction_vector[with(prediction_vector,order(-bug,loc)),]
  
  predict_x_points <- cumsum(sorted_prediction_vector$loc) # x: LOC%
  predict_Xs <- predict_x_points/predict_x_points[nrow(sorted_prediction_vector)]
  predict_y_points <- cumsum(sorted_prediction_vector$bug) # x: LOC%
  predict_Ys <- predict_y_points/predict_y_points[nrow(sorted_prediction_vector)]
  
  pos_20pc_predict <- min(which(predict_Xs >= 0.2))
  predict_x_points[pos_20pc_predict]
  
  plot(optimal_Xs,optimal_Ys,type="l")
  lines(worst_Xs,worst_Ys,type="l")
  lines(predict_Xs,predict_Ys,type="l", col = "red")
  abline(0,1)
  abline(v=.2)
  
  area_under_optimal_curve_20pc <- get20pcAUC(optimal_Xs, optimal_Ys, pos_20pc_optimal)
  area_under_prediction_curve_20pc <- get20pcAUC(predict_Xs, predict_Ys, pos_20pc_predict)
  area_under_worst_curve_20pc <- get20pcAUC(worst_Xs, worst_Ys, pos_20pc_worst)
  
  pOpt <- getPOPTNormalized(area_under_optimal_curve_20pc,area_under_prediction_curve_20pc,area_under_worst_curve_20pc)
}

#Load Data

data_size_vector<- c(1000,2000,3000,4000,5000,6000)
pOpt_vector<-c()
for(input_size in data_size_vector ){
  data<- getData("../Data/velocity_m.csv",input_size)
  train <- data$trainData
  test <-data$testData
  model<- trainFFT(train,test)
  pOpt<- calculatePOpt(test,model)
  pOpt_vector<-c(pOpt_vector,pOpt)
}

cat("pOpt vector is ",pOpt_vector)


plot(data_size_vector,pOpt_vector,type = "l",xlab = "no of records", ylab = "pOpt")
