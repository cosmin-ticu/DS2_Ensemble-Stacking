# clear memory
rm(list=ls())

library(tidyverse)
library(caret)
library(gbm)


df <- as_tibble(ISLR::Hitters) %>%
  drop_na(Salary) %>%
  mutate(log_salary = log(Salary), Salary = NULL)


# a. ----------------------------------------------------------------------

tune_grid <- expand.grid(
  .mtry = 2,
  .splitrule = "variance",
  .min.node.size = c(5, 10)
)
# random forest with 2 randomly sampled variables each time
set.seed(1234)
rf_model_2 <- train(log_salary ~ .,
                    method = "ranger",
                    data = df,
                    tuneGrid = tune_grid,
                    importance = "impurity"
)


tune_grid <- expand.grid(
  .mtry = 10,
  .splitrule = "variance",
  .min.node.size = c(5, 10)
)

# random forest with 10 randomly sampled variables each time
set.seed(1234)
rf_model_10 <- train(log_salary ~ .,
                    method = "ranger",
                    data = df,
                    tuneGrid = tune_grid,
                    importance = "impurity"
)

plot(varImp(rf_model_2), top=10) # much more balanced model as more "equal" shot are given to each variable
plot(varImp(rf_model_10), top=10) # much more aggressive at the beginning


# b. ----------------------------------------------------------------------

# explanation for the findings can be found above or in the report (in-depth)

# c. ----------------------------------------------------------------------

gbm_grid <- expand.grid(n.trees = 500, 
                        interaction.depth = 5, # maximum depth
                        shrinkage = 0.1,
                        n.minobsinnode = 5) # generic choice of 5 observations - not important in this case
set.seed(1234)
gbm_model_01_sample <- train(log_salary ~ .,
                      method = "gbm",
                      data = df,
                      bag.fraction=0.1,
                      tuneGrid = gbm_grid,
                      verbose = FALSE # gbm by default prints too much output
)

set.seed(1234)
gbm_model_full_sample <- train(log_salary ~ .,
                      method = "gbm",
                      data = df,
                      bag.fraction=1,
                      tuneGrid = gbm_grid,
                      verbose = FALSE # gbm by default prints too much output
)
plot(varImp(gbm_model_01_sample), top=10) # more balanced model
plot(varImp(gbm_model_full_sample), top=10) # more aggressive from beginning
