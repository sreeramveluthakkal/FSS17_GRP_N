library(FFTrees)

getData<- function(file1, rows){
    # orig<- read.csv(file=file1, header=TRUE, sep=",")
    # mutated<- read.csv(file=file2, header=TRUE, sep=",")
    #Remove unnecessary columns
    # bugs<-rbind(orig,mutated)
    bugs<- read.csv(file=file1, header=TRUE, sep=",")
    bugs<- bugs[-c(1,2,3)]
    # Randomize test set to test/train sections
    # https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function

    ## set the seed to make your partition reproductible
    #set.seed(123)
    train_ind <- floor(0.8 * rows)
    train <- bugs[1:train_ind, ]
    test <- bugs[(train_ind+1):rows, ]

    # cat('train_ind', train_ind, "rows", rows )

    # cat("train: ",nrow(train), "\n")
    # cat("test: ",nrow(test), "\n")
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