library(caret)
library(corrplot)
library(ggplot2)
library(tidyverse)


quantile_method <- function(x) {
  quantile1 <- quantile(x, probs = .25)
  quantile3 <- quantile(x, probs = .75)
  iqr = quantile3 - quantile1
  x > quantile3 + (iqr * 1.5) | x < quantile1 - (iqr * 1.5)
}

zscore_method <- function(x) {
  zscore <- (abs(x - mean(x)) / sd(x))
  zscore >= 3
}

remove_outliers <- function(dataframe, columns = names(dataframe), method) {
  for (col in columns) {
    if(is.numeric(dataframe[[col]]) & col != "SalePrice") {
      if(method == "quantile") {
        dataframe <- dataframe[!quantile_method(dataframe[[col]]), ]
      } else if(method == "zscore") {
        dataframe <- dataframe[!zscore_method(dataframe[[col]]), ]
      }
    }
  }
  return(dataframe)
}

train_dataset_path <- file.path("datasets", "house_prices_dataset", "train.csv")
test_dataset_path <- file.path("datasets", "house_prices_dataset", "test.csv")

train_dataset <- read.csv(train_dataset_path)
test_dataset <- read.csv(test_dataset_path)

sample_count <- nrow(train_dataset)  # for normality assumption of ols

# looking a couple of the first rows
head(train_dataset)

# print type of each column
str(train_dataset)

# adjusting types
train_dataset$MSSubClass <- as.character(train_dataset$MSSubClass)
train_dataset$OverallQual <- as.character(train_dataset$OverallQual)
train_dataset$OverallCond <- as.character(train_dataset$OverallCond)
train_dataset$YearBuilt <- as.character(train_dataset$YearBuilt)
train_dataset$YearRemodAdd <- as.character(train_dataset$YearRemodAdd)
train_dataset$GarageYrBlt <- as.character(train_dataset$GarageYrBlt)
train_dataset$MoSold <- as.character(train_dataset$MoSold)
train_dataset$YrSold <- as.character(train_dataset$YrSold)
train_dataset$YrSold <- as.character(train_dataset$YrSold)
train_dataset <- train_dataset %>% mutate(across(where(is.integer), as.numeric))

# printing descriptive statistics for each feature
summary(train_dataset)

# remove Id column
train_dataset <- select(train_dataset, -Id)

# print the number of NA values for each column
print(colSums(is.na(train_dataset)))

# drop features with more than 2/3 NA values
train_dataset <- train_dataset[, colSums(is.na(train_dataset)) <= sample_count / 3 * 2]

# replacing NA with median in numeric columns
train_dataset <- train_dataset %>% mutate(across(where(is.numeric), ~replace_na(., median(., na.rm=TRUE))))

# replacing NA with mode in categorical columns
train_dataset <- train_dataset %>% mutate(across(where(is.character), ~replace_na(., mode(.))))
train_dataset <- train_dataset %>% mutate(across(where(is.character), as.factor))

# print correlation matrix of numeric features
corr <- cor(train_dataset %>% select(where(is.numeric)), use = "pairwise.complete.obs")
highlyCorrelated <- findCorrelation(corr, cutoff = 0.75)
print(highlyCorrelated)
corr[upper.tri(corr)] <- 0
corrplot(corr, method = "circle")

# remove GarageCars because it's higly correlated to GarageArea
train_dataset <- select(train_dataset, -GarageCars)

# compare TotRmsAbvGrd vs GrLivArea
ggplot(train_dataset, aes(TotRmsAbvGrd, GrLivArea)) + geom_point()

# remove TotRmsAbvGrd because it's higly correlated to GrLivArea
train_dataset <- select(train_dataset, -TotRmsAbvGrd)

# plot stronger correlations to check homoscedasticity / heteroscedasticity assumption
ggplot(train_dataset, aes(OverallQual, SalePrice, group = OverallQual)) + geom_boxplot()
ggplot(train_dataset, aes(TotalBsmtSF, SalePrice)) + geom_point()
ggplot(train_dataset, aes(X1stFlrSF, SalePrice)) + geom_point()
ggplot(train_dataset, aes(GrLivArea, SalePrice)) + geom_point()
ggplot(train_dataset, aes(GarageArea, SalePrice)) + geom_point()

# remove outliers
train_dataset <- remove_outliers(train_dataset, method = "quantile")
print(nrow(train_dataset))

# Specify 10-fold cross validation
set.seed(42)
ctrl <- trainControl(method = "cv",  number = 10) 

# CV bagged model
bagged_cv <- train(
  SalePrice ~ .,
  data = train_dataset,
  method = "treebag",
  trControl = ctrl,
  importance = TRUE
)

# validation metrics
print(bagged_cv)

# CV liner model
linear_cv <- train(
  SalePrice ~ .,
  data = train_dataset,
  method = "lm",
  trControl = ctrl,
  importance = TRUE
)

# validation metrics
print(linear_cv)

# adjsting test dataset types
test_dataset$MSSubClass <- as.character(test_dataset$MSSubClass)
test_dataset$OverallQual <- as.character(test_dataset$OverallQual)
test_dataset$OverallCond <- as.character(test_dataset$OverallCond)
test_dataset$YearBuilt <- as.character(test_dataset$YearBuilt)
test_dataset$YearRemodAdd <- as.character(test_dataset$YearRemodAdd)
test_dataset$GarageYrBlt <- as.character(test_dataset$GarageYrBlt)
test_dataset$MoSold <- as.character(test_dataset$MoSold)
test_dataset$YrSold <- as.character(test_dataset$YrSold)
test_dataset$YrSold <- as.character(test_dataset$YrSold)
test_dataset <- train_dataset %>% mutate(across(where(is.integer), as.numeric))

# final test evaluation
pred <- predict(bagged_cv, test_dataset)
MAE(pred, test_dataset$SalePrice)

# checking how the error is distributed
results <- data.frame(pred = pred, target = test_dataset$SalePrice, error = abs(pred - test_dataset$SalePrice))
ggplot(results, aes(pred, target)) + geom_point() + geom_abline(0, slope = 1, color = "red")

# plotting errors distribution
ggplot(results, aes(error)) + geom_density(aes(y = ..count..), fill = "lightgray") +
  geom_vline(aes(xintercept = mean(error)), 
             linetype = "dashed", size = 0.6,
             color = "red")

# compute metrics for Kaggle submission
print(RMSE(log(results$pred), log(results$target)))
