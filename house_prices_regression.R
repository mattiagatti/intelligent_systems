library(caret)
library(corrplot)
library(tidyverse)

cutoff_redundant <- 0.5
cutoff_useless <- 0.1
top_categ <- 15

# returns the most useful numeric features
filter_quantitative_features <- function(df) {
  df <- df %>% select(where(is.numeric))
  corr <- cor(df, use = "pairwise.complete.obs", method = "pearson")
  
  highlyCorrelated <- findCorrelation(corr, cutoff = cutoff_redundant)
  corr[upper.tri(corr)] <- 0
  corrplot(corr, method = "circle")
  
  lastRow <- corr[nrow(corr), ]
  noCorrelated <- which(lastRow < cutoff_useless, arr.ind = TRUE)
  worst <- union(highlyCorrelated, noCorrelated)
  worst <- colnames(df)[worst]
  worst <- worst[!worst == 'SalePrice']

  return (worst)
}

# returns the most useful categorical features
filter_categorical_features <- function(df) {
  decision <- df$SalePrice
  predictors <- df %>% select(where(is.factor))
  randomForest <- caret::train(predictors, decision, data = df,
                               method = "rf", ntree = 100, importance = TRUE)
  imp <- varImp(randomForest, scale = TRUE)
  imp <- imp$importance
  plot(imp)
  best <- rownames(imp)[order(imp$Overall, decreasing=TRUE)[1:top_categ]]
  worst <- setdiff(rownames(imp), best)

  return (worst)
}

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

set.seed(42)

train_dataset_path <- file.path("datasets", "house_prices_dataset", "train.csv")
test_dataset_path <- file.path("datasets", "house_prices_dataset", "test.csv")

train_dataset <- read.csv2(train_dataset_path, sep=",", stringsAsFactors = TRUE)
test_dataset <- read.csv2(test_dataset_path, sep=",", stringsAsFactors = TRUE)

# merging the two splits for easier preprocessing
test_dataset$SalePrice <- 0
train_length <- nrow(train_dataset)
dataset <- rbind(train_dataset, test_dataset)

dataset <- select(dataset, -Id)

# adjusting types
dataset$MSSubClass <- as.factor(dataset$MSSubClass)
dataset$OverallQual <- as.factor(dataset$OverallQual)
dataset$OverallCond <- as.factor(dataset$OverallCond)

### DATA EXPLORATION ###
dataset %>%
  keep(is.numeric) %>%   
  gather() %>%                  
  ggplot(aes(value)) + 
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

dataset %>%
  keep(is.factor) %>%
  gather() %>%
  ggplot(aes(value)) + 
  facet_wrap(~ key, scales = "free") +
  geom_bar()

# first removals
dataset <- select(dataset, -GarageCars)
dataset <- select(dataset, -GarageYrBlt)
dataset <- select(dataset, -Utilities)

# print type of each column with missing values
na_cols <- which(colSums(is.na(dataset)) > 0)
str(dataset[na_cols])

### handling numeric types missing values ###

# here NA means the feature is missing in the dataset (not recorded)
cols <- c("GarageCars", "GarageArea", "BsmtFinSF1")
dataset <- dataset %>% mutate(across(where(is.numeric), ~replace_na(., median(.))))

# in the remaining numeric features NA means 0
dataset <- dataset %>% mutate(across(where(is.numeric), ~replace_na(., 0)))

### handling factor types missing values ###

# here NA means the feature is missing in the dataset (not recorded)
cols <- c("MSZoning", "Exterior1st", "Exterior2nd", "MasVnrType", "Electrical", "KitchenQual", "Functional", "SaleType")
dataset <- dataset %>% mutate(across(all_of(cols), ~replace_na(., mode(.))))

# here NA means the feature is missing in the house
cols <- c("Alley", "BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtCond")
dataset <- dataset %>% mutate(across(all_of(cols), ~replace_na(., "NO")))

# convert all char columns to factors (they are categorical variables)
dataset <- dataset %>% mutate(across(where(is.character), as.factor))

# printing descriptive statistics for each feature
summary(dataset)

# remove near zero variance features
dataset <- dataset[,-nearZeroVar(dataset)]

# rank quantitative features
worst_num <- filter_quantitative_features(dataset[1:train_length,])
dataset <- select(dataset, -worst_num)

# rank categorical features
worst_cat <- filter_categorical_features(dataset[1:train_length,])
dataset <- select(dataset, -worst_cat)

### MODEL ###
train_dataset <- dataset[1:train_length,]

# remove outliers
# train_dataset <- remove_outliers(train_dataset, method = "quantile")

test_dataset <- dataset[train_length + 1:nrow(dataset),]

# Specify 10-fold cross validation
ctrl <- trainControl(method = "cv",  number = 10) 

# CV bagged model
bagged_model <- caret::train(
  SalePrice ~ .,
  data = train_dataset,
  method = "rf",
  trControl = ctrl
)

# validation metrics
print(bagged_model)

linear_model <- caret::train(
  SalePrice ~ .,
  data = train_dataset,
  method = 'lm',
  trControl = ctrl
)

# validation metrics
print(linear_model)

