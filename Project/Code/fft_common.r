library(FFTrees)

getData<- function(filename, rows){
    bugs<- read.csv(file=filename, header=TRUE, sep=",", nrows=rows)
    #Remove unnecessary columns
    bugs<- bugs[-c(1,2,3)]
    # Randomize test set to test/train sections
    # https://stackoverflow.com/questions/17200114/how-to-split-data-into-training-testing-sets-using-sample-function
    smp_size <- floor(0.8 * nrow(bugs))
    ## set the seed to make your partition reproductible
    set.seed(123)
    train_ind <- sample(seq_len(nrow(bugs)), size = smp_size)
    train <- bugs[train_ind, ]
    test <- bugs[-train_ind, ]
    output <- list("trainData" = train, "testData" = test)
}

trainFFT<- function(test, train){
    model <- FFTrees(formula = bug ~ wmc+dit+noc+cbo+rfc+lcom+ca+ce+npm+lcom3+loc+moa+mfa+cam+ic+cbm+amc+max_cc+avg_cc,
                    data = train,                
                    data.test = test,        
                    main = "Bug Detector",          
                    decision.labels = c("No Bug", "Bug"))
}