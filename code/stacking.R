# clear memory
rm(list=ls())

library(tidyverse)
library(h2o)

# h2o.shutdown()
h2o.init(min_mem_size = '4g' ,max_mem_size = '8g')
data3 <- read_csv("data/KaggleV2-May-2016.csv")

# some data cleaning
data3 <- select(data3, -one_of(c("PatientId", "AppointmentID", "Neighbourhood"))) %>%
  janitor::clean_names()

# for binary prediction, the target variable must be a factor + generate new variables
data3 <- mutate(
  data3,
  no_show = factor(no_show, levels = c("Yes", "No")),
  handcap = ifelse(handcap > 0, 1, 0),
  across(c(gender, scholarship, hipertension, alcoholism, handcap, diabetes), factor),
  hours_since_scheduled = as.numeric(appointment_day - scheduled_day)
)

# clean up a little bit
data3 <- filter(data3, between(age, 0, 95), hours_since_scheduled >= 0) %>%
  select(-one_of(c("scheduled_day", "appointment_day", "sms_received")))

data3 <- as.h2o(data3)

# Peek at the data
skimr::skim(data3)

# a. ----------------------------------------------------------------------

## Create train / validation / test sets, cutting the data into 5% - 45% - 50% parts.
splitted_data <- h2o.splitFrame(data3, 
                                ratios = c(0.05, 0.5), 
                                seed = 123)
str(splitted_data)
data_train <- splitted_data[[1]]
data_validation <- splitted_data[[2]]
data_test <- splitted_data[[3]]


# b. ----------------------------------------------------------------------

y <- "no_show"
X <- setdiff(names(data3), y)

rf_model_benchmark <- h2o.randomForest(
  X, y,
  training_frame = data_train,
  ntrees = 500, 
  mtries = 2,
  seed = 123,
  nfolds = 5)

# xval stands for 'cross-validation'
print(h2o.performance(rf_model_benchmark, xval = TRUE))

rf_benchmark<- h2o.auc(h2o.performance(rf_model_benchmark, newdata = data_validation))
rf_benchmark


# c. ----------------------------------------------------------------------

## GLM - Ridge/LASSO/Elastic Net

glm_parameters <- list(alpha = c(0, .25, .5, .75, 1))

# build grid search with previously selected hyperparameters
glm_grid <- h2o.grid(
  "glm", x = X, y = y,
  training_frame = data_train,
  lambda_search = TRUE,   # performs search for optimal lambda as well
  nfolds = 5,
  seed = 123,
  hyper_params = glm_parameters, 
  keep_cross_validation_predictions = TRUE)

glm_grid # difference between Ridge & LASSO is very small

# get the best model based on the cross-validation exercise
glm_model <- h2o.getModel(glm_grid@model_ids[[1]])
glm_model

## GBM

gbm_parameters <- list(ntrees = c(100,500),
                       max_depth = c(2,5), # going from simple trees to the default value
                       learn_rate = c(.01,.1)) # tuning aggressivity

# build grid search with previously selected hyperparameters
gbm_grid <- h2o.grid(
  "gbm", x = X, y = y,
  training_frame = data_train,
  nfolds = 5,
  seed = 123,
  hyper_params = gbm_parameters, 
  keep_cross_validation_predictions = TRUE)

gbm_grid

# get the best model based on the cross-validation exercise
gbm_model <- h2o.getModel(gbm_grid@model_ids[[1]])
gbm_model

## Random Forest

rf_parameters <- list(ntrees = c(100,500),
                      mtries = c(2,4))

rf_grid <- h2o.grid(x = X, 
                    y = y, 
                    training_frame = data_train, 
                    algorithm = "randomForest", 
                    nfolds = 5,
                    seed = 123,
                    hyper_params = rf_parameters, 
                    keep_cross_validation_predictions = TRUE)

# get the best model based on the cross-validation exercise
rf_model <- h2o.getModel(rf_grid@model_ids[[1]])
rf_model

## Deeplearning model attempt

deeplearning_model <- h2o.deeplearning(
  X, y,
  training_frame = data_train,
  seed = 123,
  nfolds = 5, 
  keep_cross_validation_predictions = TRUE
)


# d. ----------------------------------------------------------------------

# predict on validation set
validation_performances <- list(
  "glm" = h2o.auc(h2o.performance(glm_model, newdata = data_validation)),
  "rf" = h2o.auc(h2o.performance(rf_model, newdata = data_validation)),
  "gbm" = h2o.auc(h2o.performance(gbm_model, newdata = data_validation)),
  "deeplearning" = h2o.auc(h2o.performance(deeplearning_model, newdata = data_validation))
)

# look at AUC for all models
validation_performances

# Nicer looking comparison
my_models <- list(
  glm_model, gbm_model, rf_model, deeplearning_model)

auc_on_validation <- map_df(my_models, ~{
  tibble(model = .@model_id, AUC = h2o.auc(h2o.performance(., data_validation)))
}) %>% arrange(AUC)


# e. ----------------------------------------------------------------------

h2o.model_correlation_heatmap(my_models, data_validation)
h2o.varimp_heatmap(my_models)


# f. ----------------------------------------------------------------------


