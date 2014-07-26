setwd("/Users/MacBook/Desktop/David/COURSERA/Data Science/Practical Machine Learning")
install.packages("caret")

library(caret)
library(RANN)

data <- read.csv("pml-training.csv")
testing2 <- read.csv("pml-testing.csv")

inTrain <- createDataPartition(y=data$classe,p=0.75,list=FALSE)

training <- data[inTrain,]
testing <- data[-inTrain,]

nzv <- nearZeroVar(training,saveMetrics=TRUE)
nzvCols <- nzv$nzv

training <- training[,!nzvCols]
testing <- testing[,!nzvCols]
testing2 <- testing2[,!nzvCols]

n <- ncol(training)
m <- ncol(training)

lnumeric <- sapply(training,is.numeric)

numtrain <- training[,lnumeric]
numtest <- testing[,lnumeric]
numtest2 <- testing2[,lnumeric]

naID <- rep(TRUE,ncol(numtrain))

for(i in 1:ncol(numtrain))
{
    if(sum(is.na(numtrain[,i])) > 0)
    {
        naID[i] <- FALSE
    }
}

numtrain <- numtrain[,naID]
numtest <- numtest[,naID]
numtest2 <- numtest2[,naID]

#pp <- preProcess(numtrain,method="pca",thresh=0.9)
#numtrain <- predict(pp,numtrain)
#numtest <- predict(pp,numtest)

modelFit <- train(training[,m] ~ ., method="rf", preProcess="pca", trControl=trainControl(method="cv"),data=numtrain)

confusionMatrix(testing[,m],predict(modelFit,numtest))

pred <- predict(modelFit,numtest2)

for(i in 1:length(pred))
{
    filename <- paste0("problem_id_",i,".txt")
    write.table(pred[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
}