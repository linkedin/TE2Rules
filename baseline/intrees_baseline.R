###
# R script to create a tree ensemble model (with given ntrees, max_depth) and explain
# it using inTrees. The script reads training and testing data and then writes the output
# of the tree ensemble model (scores, class_predictions) and rules (extracted rules,
# class_predictions) on both training and testing data in the results directory.

# Usage: Rscript intrees_baseline.R train_file, test_file, result_dir, n_estimators, max_depth
###

# args
train_file = commandArgs(trailingOnly=TRUE)[1]
test_file = commandArgs(trailingOnly=TRUE)[2]
res_dir = commandArgs(trailingOnly=TRUE)[3]
ntree = as.integer(commandArgs(trailingOnly=TRUE)[4])
max_depth = as.integer(commandArgs(trailingOnly=TRUE)[5])

# library
if (!require(xgboost)) install.packages('xgboost')
if (!require(inTrees)) install.packages('inTrees')
library(xgboost)
library(inTrees)
set.seed(123)

# data
data_train <- read.csv(train_file)
X_train <- as.matrix(data_train[,1:(ncol(data_train)-1)])
y_train <- data_train$label_1

data_test <- read.csv(test_file)
X_test <- as.matrix(data_test[,1:(ncol(data_test)-1)])
y_test <- data_test$label_1

# xgb training
xgb <- xgboost(data = X_train, label = y_train,
               nround = ntree, max_depth = max_depth,
               objective = "binary:logistic")

# build rules from inTrees
treeList <- XGB2List(xgb, X_train)
ruleExec <- unique(extractRules(treeList, X_train))

# evaluate on data and use it to prune (optional: pruning gives less fidelity, smaller rules)
ruleMetric <- getRuleMetric(ruleExec, X_train, y_train)
ruleMetric <- unique(pruneRule(ruleMetric, X_train, y_train))

# build a if-else program: only works for classifiaction. XGB in R is a regression.
# ruleOrdered <- buildLearner(ruleMetric, X_train, y_train)
ruleOrdered <- ruleMetric
y_pred_train_rules <- applyLearner(ruleOrdered, X_train)
y_pred_train_rules <- as.numeric(y_pred_train_rules > 0.5)
y_pred_test_rules <- applyLearner(ruleOrdered, X_test)
y_pred_test_rules <- as.numeric(y_pred_test_rules > 0.5)

# save
res_dir = sprintf('%s/intrees', res_dir)
dir.create(res_dir, showWarnings = FALSE)

y_pred_train <- predict(xgb, as.matrix(X_train))
y_pred_test <- predict(xgb, as.matrix(X_test))

write.table(y_pred_train, sprintf('%s/pred_train_score.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)
write.table(y_pred_test, sprintf('%s/pred_test_score.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)

y_pred_train <- as.numeric(y_pred_train > 0.5)
y_pred_test <- as.numeric(y_pred_test > 0.5)

write.table(y_pred_train, sprintf('%s/pred_train.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)
write.table(y_pred_test, sprintf('%s/pred_test.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)
write.table(y_pred_train_rules, sprintf('%s/pred_train_rules.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)
write.table(y_pred_test_rules, sprintf('%s/pred_test_rules.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)

explanation <- presentRules(ruleOrdered,colnames(X_train))
write.table(explanation, file=sprintf("%s/rules.txt", res_dir), quote=F, sep=", ", row.names=F, append=F)
