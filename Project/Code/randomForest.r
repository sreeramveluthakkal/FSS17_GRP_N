# Load the party package. It will automatically load other required packages.
# library(party)
library(randomForest)
source('fft_common.r')
args <- commandArgs(TRUE)
n <- as.double(args[1])
dataset_name <- args[2]

# Read data
# bugs<- read.csv(file="../Data/velocity.csv", header=TRUE, sep=",", nrows=n)
# #Remove unnecessary columns
# bugs<- bugs[-c(1,2,3)]

# # Randomize test set to test/train sections
# # https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
# smp_size <- floor(0.8 * nrow(bugs))

# # set the seed to make your partition reproductible
# set.seed(123)
# train_ind <- sample(seq_len(nrow(bugs)), size = smp_size)
# train <- bugs[train_ind, ]
# test <- bugs[-train_ind, ]

# data<- getData("../Data/velocity.csv", "../Data/velocity_m.csv", n)
data<- getData(paste("../Data/", dataset_name, "_m.csv", sep=""), n)

train <- data$trainData
test <-data$testData

# Saving the training and testing set to file:
write.csv(train, file = paste('../Results/', dataset_name, "_", toString(n), "_training.csv", sep=""))
write.csv(test, file = paste('../Results/', dataset_name, "_", toString(n), "_testing.csv", sep=""))

# train the model
model <- randomForest(formula = factor(bug) ~ wmc+dit+noc+cbo+rfc+lcom+ca+ce+npm+lcom3+loc+moa+mfa+cam+ic+cbm+amc+max_cc+avg_cc,
                        data = train,                
                        test = test,
                        ntree=1000,
                        nodesize=25,
                        main = "Bug Detector", 
                        decision.labels = c("No Bug", "Bug"),
                        importance=TRUE)

# get prediction for the test set
prediction <- predict(model, data = test)

# cat('=============\n')
# cat('prediction', prediction)
# cat('prediction size', length(prediction))
# cat('test size', nrow(test))

# compare prediction and actual, counting no of hits:
count <- 0
for(i in 1:min(length(prediction), nrow(test))){
  # cat("=================\n")
  # cat(i, "\n")
  # cat("prediction[i] =", prediction[i], " \n")
  # cat("test[i, 'bug'] =", test[i, "bug"], " \n")
  if(test[i,"bug"]==prediction[i]){
    count = count +1;
  }
}

# # View the forest results.

# saving model to file
sink(paste('../Results/', dataset_name, "_", toString(n), "_rf.txt", sep=""))
print(model)
sink()

# print(model) 

cat("number of correct prediction", count, "\n")
cat("model size", object.size(model), "\n")

# # Importance of each predictor.
# print(importance(model,type = 2))


