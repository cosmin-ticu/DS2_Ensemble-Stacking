# clear memory
rm(list=ls())

library(tidyverse)
library(caret)
library(rpart.plot)
library(gbm)
library(xgboost)
library(caTools)
library(pROC)
library(viridis)

data <- as_tibble(ISLR::OJ)

table(data$Purchase)


# a. ----------------------------------------------------------------------


# Work vs holdout sets
set.seed(1234)
train_indices <- as.integer(createDataPartition(data$Purchase, 
                                                p = 0.75, list = FALSE))
data_train <- data[train_indices, ]
data_holdout <- data[-train_indices, ]

fitControl <- trainControl(method = "cv", number = 5,
                           summaryFunction = twoClassSummary,
                           classProbs=TRUE,
                           verboseIter = TRUE,
                           savePredictions = TRUE)
set.seed(1234)
model_tree_benchmark<-train(
  Purchase~.,
  data=data_train,
  method="rpart",
  trControl=fitControl
)
model_tree_benchmark

plot(model_tree_benchmark)

rpart.plot(model_tree_benchmark$finalModel)


# b. ----------------------------------------------------------------------


## Probability Forest

train_control <- trainControl(
  method = "cv",
  n = 5,
  classProbs = TRUE, # same as probability = TRUE in ranger
  summaryFunction = twoClassSummary,
  savePredictions = TRUE,
  verboseIter = F
)


tune_grid <- expand.grid(
  .mtry = c(2,3,4,5),
  .splitrule = "gini",
  .min.node.size = c(5, 10)
)


# random forest
set.seed(1234)
model_rf <- train(Purchase~ .,
                  data = data_train,
                  method = "ranger",
                  trControl = train_control,
                  tuneGrid = tune_grid,
                  importance = "impurity"
)
model_rf


## Gradient Boosting Machine

gbmGrid <-  expand.grid(interaction.depth = c(1, 3, 5, 7, 9), 
                        n.trees = 500, # known standard for GBM (stabilizes later than RF)
                        shrinkage = c(0.005, 0.01),
                        n.minobsinnode = c(2,5)) # to check overfitting

set.seed(1234)
model_gbm <- train(Purchase~ .,
                   data = data_train,
                   method = "gbm",
                   trControl = train_control,
                   tuneGrid = gbmGrid,
                   verbose = F
)
model_gbm # 500 trees, 5 nodes, 0.01 shrinkage and 2 min observations per tree


## XGBoost

xgb_grid <- expand.grid(nrounds = 500,
                        max_depth = c(1, 3, 5, 7, 9),
                        eta = c(0.005, 0.01),
                        gamma = 0.01,
                        colsample_bytree = c(0.3, 0.5, 0.7),
                        min_child_weight = 1, # similar to n.minobsinnode
                        subsample = c(0.5))
set.seed(1234)
model_xgboost <- train(Purchase ~ .,
                       method = "xgbTree",
                       data = data_train,
                       trControl = train_control,
                       tuneGrid = xgb_grid)
model_xgboost


# c. ----------------------------------------------------------------------

resamples <- resamples(list("decision_tree_benchmark" = model_tree_benchmark,
                            "rf" = model_rf,
                            "gbm" = model_gbm,
                            "xgboost" = model_xgboost))
summary(resamples)

logit_models <- list()
logit_models[["decision_tree_benchmark"]] <- model_tree_benchmark
logit_models[["RF"]] <- model_rf
logit_models[["GBM"]] <- model_gbm
logit_models[["XGBoost"]] <- model_xgboost

CV_AUC_folds <- list()

for (model_name in names(logit_models)) {
  
  auc <- list()
  model <- logit_models[[model_name]]
  for (fold in c("Fold1", "Fold2", "Fold3", "Fold4", "Fold5")) {
    cv_fold <-
      model$pred %>%
      filter(Resample == fold)
    
    roc_obj <- roc(cv_fold$obs, cv_fold$CH)
    auc[[fold]] <- as.numeric(roc_obj$auc)
  }
  
  CV_AUC_folds[[model_name]] <- data.frame("Resample" = names(auc),
                                           "AUC" = unlist(auc))
}

CV_AUC <- list()

for (model_name in names(logit_models)) {
  CV_AUC[[model_name]] <- mean(CV_AUC_folds[[model_name]]$AUC)
}

# d. ----------------------------------------------------------------------

## ROC Plot with built-in package 
gbm_pred<-predict(model_gbm, data_holdout, type="prob")
colAUC(gbm_pred, data_holdout$Purchase, plotROC = TRUE)

## ROC plot with own function
data_holdout[,"best_model_no_loss_pred"] <- gbm_pred[,"CH"]

roc_obj_holdout <- roc(data_holdout$Purchase, data_holdout$best_model_no_loss_pred)

createRocPlot <- function(r, plot_name) {
  all_coords <- coords(r, x="all", ret="all", transpose = FALSE)
  
  roc_plot <- ggplot(data = all_coords, aes(x = fpr, y = tpr)) +
    geom_line(color='blue', size = 0.7) +
    geom_area(aes(fill = 'red', alpha=0.4), alpha = 0.3, position = 'identity', color = 'blue') +
    scale_fill_viridis(discrete = TRUE, begin=0.6, alpha=0.5, guide = FALSE) +
    xlab("False Positive Rate (1-Specifity)") +
    ylab("True Positive Rate (Sensitivity)") +
    geom_abline(intercept = 0, slope = 1,  linetype = "dotted", col = "black") +
    scale_y_continuous(limits = c(0, 1), breaks = seq(0, 1, .1), expand = c(0, 0.01)) +
    scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .1), expand = c(0.01, 0)) + 
    theme_bw()

  roc_plot
}

createRocPlot(roc_obj_holdout, "ROC curve for best model (GBM)")


# e. ----------------------------------------------------------------------

plot(varImp(model_rf), top = 5)

plot(varImp(model_gbm), top = 5)

plot(varImp(model_xgboost), top = 5)

