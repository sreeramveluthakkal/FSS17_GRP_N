#Load Data
df <- read.csv(file="/Users/aswinak/Documents/CSC591/FinalProject/FSS17_GRP_N/Project/Data/velocity_m.csv", header=TRUE, sep=",")
#Remove unnecessary columns
df2<- df[-c(1,2,3)]

train_size <- floor(0.8 * nrow(df2))
#Split first set to test and train data
set1 <- df2[1:train_size,]  # for training
set2 <- df2[(train_size+1):nrow(df2),]  # for testing

#Map 0/1 to TRUE/FALSE in the bug field
#set2$bug<- set2$bug>0


#Randomize test set to test/train sections
smp_size <- floor(0.8 * nrow(set1))
## set the seed to make your partition reproductible
set.seed(123)
train_ind <- sample(seq_len(nrow(set1)), size = smp_size)
train_data <- set1[train_ind, ]
test_data <- set1[-train_ind, ]

model <- FFTrees(formula = bug ~ wmc + dit + noc +cbo+ rfc+ lcom+ca+ce+npm+lcom3+loc+moa+mfa+cam+ic+cbm+amc+max_cc+avg_cc,         
                 data = train_data,                
                 data.test = test_data,         
                 main = "Bug Detector",          
                 decision.labels = c("No Bug", "Bug"))


getAUC<- function(x_axis,y_axis){
  id <- order(x_axis)
  sum(diff(x_axis[id])*rollmean(y_axis[id],2))
}

#https://stackoverflow.com/questions/4954507/calculate-the-area-under-a-curve
get20pcAUC<- function(x_axis,y_axis,cutoff){
  x_axis <- x_axis[1:cutoff]
  y_axis <- y_axis[1:cutoff]
  print(y_axis)
  id <- order(x_axis)
  sum(diff(x_axis[id])*rollmean(y_axis[id],2))
}

getPOPTNormalized <- function(area_optimal, area_model, area_worst){
  1 - ( (area_optimal - area_model)/(area_optimal - area_worst))
}

calculatePOpt<- function(dataSet, model){
  
  prediction <- predict(model, data = set2)
  
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
  for (i in 1: nrow(set2)){
    prediction_vector[i,] <- c(set2[i,"loc"],if (prediction[i]==TRUE)1 else 0 )
  }
  df[with(df,order(-bug,loc)),]
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

pOpt1<- calculatePOpt(set2,model)

cat("pOpt is ",pOpt1)
