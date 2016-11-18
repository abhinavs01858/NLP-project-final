setwd("C:/Final year/Projects/NLP")

library(data.table)
library(xgboost)
train <- fread("training2.csv")
test <- fread("testing2.csv")


# train$V14 <- as.numeric(as.factor(train$V14))-1
target <- train$V14
train[,V14:=NULL]
solution <- test$V14
test[,V14:=NULL]


train <- sapply(train,as.numeric)
test <- sapply(test, as.numeric)
# train <- as.matrix(train)
# test <- as.matrix(test)



train.xg = xgb.DMatrix(train, label=target)
#test.xg = xgb.DMatrix(test)
param <- list(max_depth = 10,
              eta = 0.1,
              objective="reg:logistic",
              subsample = 0.9,
              eval_metric = "merror",
              min_child_weight = 4,
              colsample_bytree = 0.9
              
)
set.seed(1)
start_time <- Sys.time()
model_xgb2 <- xgb.train(param, train.xg, nthread = 16, nround = 1000,verbose = 1)
end_time <- Sys.time()
time_taken <- end_time - start_time

prediction<-as.data.table(predict(model_xgb2,test))
prediction$V1[prediction$V1 > 0.78 ] <- 1
prediction$V1[prediction$V1 < 1] <- 0


prediction$ans <- solution



length(which(prediction$V1 == prediction$ans))




#240 correct preditions













#multisoftprob




train.xg = xgb.DMatrix(train, label=target)
#test.xg = xgb.DMatrix(test)
param <- list(max_depth = 10,
              eta = 0.1,
              objective="multi:softprob",
              num_class = 2,
              subsample = 0.9,
              eval_metric = "merror",
              min_child_weight = 4,
              colsample_bytree = 0.9
              
)
set.seed(1)
start_time <- Sys.time()
model_xgb2 <- xgb.train(param, train.xg, nthread = 16, nround = 1000,verbose = 1)
end_time <- Sys.time()
time_taken <- end_time - start_time

prediction<-as.data.table(predict(model_xgb2,test))

probab <- copy(test)
probab <- as.data.table(probab)
probab$fact <- prediction$V1[1:nrow(probab)]
probab$opi <- prediction$V1[319:636]


probab <- as.data.frame(probab)
probab <- probab[-c(1:13)]
probab$pred <- 1 
probab$pred[probab$fact > probab$opi] <- 0

#160 correct predictions
