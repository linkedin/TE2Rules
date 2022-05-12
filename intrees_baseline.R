# args
train_file = commandArgs(trailingOnly=TRUE)[1]
res_dir = commandArgs(trailingOnly=TRUE)[2]

# library
library(randomForest)
library(inTrees)
set.seed(123)

# data
data_train <- read.csv(train_file)
X_train <- data_train[,1:(ncol(data_train)-1)]
y_train <- as.factor(data_train[,ncol(data_train)])

# rf training
ntree <- 10
rf <- randomForest(X_train, y_train, ntree=ntree, nodesize=10, maxnodes=8)
err = rf$err.rate[ntree]
print(err)

# build rules from inTrees
treeList <- RF2List(rf)
ruleExec <- unique(extractRules(treeList, X_train))

# evaluate on data and use it to prune
ruleMetric <- getRuleMetric(ruleExec, X_train, y_train)
ruleMetric <- unique(pruneRule(ruleMetric, X_train, y_train))

# build a if-else program
ruleOrdered <- buildLearner(ruleMetric, X_train, y_train)
y_pred_train_rules <- applyLearner(ruleOrdered, X_train)

# save
res_dir = sprintf('%s/intrees', res_dir)
dir.create(res_dir, showWarnings = FALSE)

y_pred_train <- predict(rf, X_train)
write.table(y_pred_train, sprintf('%s/pred_train.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)
write.table(y_pred_train_rules, sprintf('%s/pred_train_rules.csv', res_dir), quote=F, row.names=F, col.names=F, append=F)

explanation <- presentRules(ruleOrdered,colnames(X_train))
write.table(explanation, file=sprintf("%s/rules.txt", res_dir), quote=F, sep=", ", row.names=F, append=F)

forest_dir = sprintf('%s/forest', res_dir)
dir.create(forest_dir, showWarnings = FALSE)

for (t in 1:ntree) {
    tree <- getTree(rf, k=t)
    write.table(tree, file=sprintf("%s/tree%03d.txt", forest_dir, t), quote=F, sep=", ", row.names=F, append=F)
}


