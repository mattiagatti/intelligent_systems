library(caret)
library(corrplot)
library(ggplot2)
library(tidyverse)

cutoff_collinear <- 0.8
cutoff_useless <- 0.2
top_categ <- 10

# returns the most useful numeric features
filter_quantitative_features <- function(df) {
  df <- df %>% select(where(is.numeric))
  corr <- cor(df, use = "pairwise.complete.obs", method = "pearson")
  
  highlyCorrelated <- findCorrelation(corr, cutoff = cutoff_collinear)
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
  var_imp_filename <- "var_imp.csv"
  if (!file.exists(var_imp_filename)) {
    decision <- df$SalePrice
    predictors <- df %>% select(where(is.factor))
    randomForest <- caret::train(predictors, decision, data = df,
                                 method = "rf", importance = TRUE)
    imp <- varImp(randomForest)
    write.csv(imp$importance, file = "var_imp.csv")
  }
  
  imp <- read.csv('var_imp.csv', row.names = 1)

  best <- rownames(imp)[order(imp$Overall, decreasing=TRUE)[1:top_categ]]
  worst <- setdiff(rownames(imp), best)

  return (worst)
}

set.seed(42)

train_dataset_path <- file.path("datasets", "house_prices_dataset", "train.csv")
test_dataset_path <- file.path("datasets", "house_prices_dataset", "test.csv")

train_dataset <- read.csv2(train_dataset_path, sep=",", stringsAsFactors = TRUE)
test_dataset <- read.csv2(test_dataset_path, sep=",", stringsAsFactors = TRUE)

# merging the two splits for easier preprocessing
test_dataset$SalePrice <- 0
tl <- nrow(train_dataset)
dataset <- rbind(train_dataset, test_dataset)

dataset <- select(dataset, -Id)

# adjusting types
dataset$MSSubClass <- as.factor(dataset$MSSubClass)
dataset$OverallQual <- as.factor(dataset$OverallQual)
dataset$OverallCond <- as.factor(dataset$OverallCond)

### DATA EXPLORATION ###
dataset[1:tl,] %>%
  keep(is.numeric) %>%   
  gather() %>%                  
  ggplot(aes(value)) + 
  facet_wrap(~ key, scales = "free") +
  geom_histogram()

dataset[1:tl,] %>%
  keep(is.factor) %>%
  gather() %>%
  ggplot(aes(value)) + 
  facet_wrap(~ key, scales = "free") +
  geom_bar()

# first removals
dataset <- select(dataset, -GarageCars)
dataset <- select(dataset, -GarageYrBlt)
dataset <- select(dataset, -Utilities)
dataset <- select(dataset, -Exterior2nd)

# print type of each column with missing values
na_cols <- which(colSums(is.na(dataset[1:tl,])) > 0)
str(dataset[1:tl, na_cols])

# drop features with more than 3/4 NA values
# dataset <- dataset[, colSums(is.na(dataset[1:tl,])) <= tl / 4 * 3]

### handling numeric types missing values ###

# here NA means the feature is missing in the dataset (not recorded)
cols <- c("GarageCars", "GarageArea", "BsmtFinSF1")
dataset <- dataset %>% mutate(across(where(is.numeric), ~replace_na(., median(.))))

# in the remaining numeric features NA means 0
dataset <- dataset %>% mutate(across(where(is.numeric), ~replace_na(., 0)))

### handling factor types missing values ###

dataset <- dataset %>% mutate(across(where(is.factor), as.character))

# here NA means the feature is missing in the dataset (not recorded)
# cols <- c("MSZoning", "Exterior1st", "Exterior2nd", "MasVnrType", "Electrical", "KitchenQual", "Functional", "SaleType")
cols <- c("MSZoning", "Exterior1st", "MasVnrType", "Electrical", "KitchenQual", "Functional", "SaleType")
dataset <- dataset %>% mutate(across(all_of(cols), ~replace_na(., mode(.))))

# here NA means the feature is missing in the house
cols <- c("Alley", "BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "PoolQC", "Fence", "MiscFeature", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtCond")
# cols <- c("BsmtQual", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "FireplaceQu", "GarageType", "GarageFinish", "GarageQual", "GarageCond", "BsmtCond")
dataset <- dataset %>% mutate(across(all_of(cols), ~replace_na(., "NO")))

# convert all char columns to factors (they are categorical variables)
dataset <- dataset %>% mutate(across(where(is.character), as.factor))

# printing descriptive statistics for each feature
summary(dataset[1:tl,])

# remove near zero variance features
dataset <- dataset[,-nearZeroVar(dataset[1:tl,])]

# rank quantitative features
worst_num <- filter_quantitative_features(dataset[1:tl,])
dataset <- select(dataset, -worst_num)

# rank categorical features
worst_cat <- filter_categorical_features(dataset[1:tl,])
dataset <- select(dataset, -worst_cat)

train_dataset <- dataset[1:tl,]

### OUTLIERS REMOVAL ###

# plot all scatterplots
values <- train_dataset %>% keep(is.numeric) %>% select(-SalePrice)
times <- ncol(values)
values <- values %>% gather()
values <- cbind(values, rep(train_dataset$SalePrice, times = times))
colnames(values)[3] <- "SalePrice"

ggplot(data = values, aes(x = value, y = SalePrice)) + 
  facet_wrap(~ key, scales = "free") +
  geom_point(size=2, shape=20)

# plot overallcond
ggplot(train_dataset, aes(x = factor(OverallQual,level=1:10), y = SalePrice,
                          fill = OverallQual)) + geom_boxplot() +
  scale_x_discrete(name ="OverallQual") +
  scale_fill_discrete(breaks=1:10)

# fix thresholds
train_dataset <- train_dataset[train_dataset$BsmtFinSF1 < 2000, ]
train_dataset <- train_dataset[train_dataset$BsmtFullBath < 3, ]
train_dataset <- train_dataset[train_dataset$GarageArea < 1250, ]
train_dataset <- train_dataset[train_dataset$LotArea < 100000, ]
train_dataset <- train_dataset[train_dataset$LotFrontage < 200, ]
train_dataset <- train_dataset[train_dataset$MasVnrArea < 1500, ]
train_dataset <- train_dataset[train_dataset$TotalBsmtSF < 3000, ]

### MODEL ###

# Specify 10-fold cross validation
ctrl <- trainControl(method = "cv",  number = 10, allowParallel = TRUE) 

# CV bagged model
bagged_model <- caret::train(
  SalePrice ~ .,  # log(SalePrice) ~ .
  data = train_dataset,
  method = "rf",
  trControl = ctrl
)

linear_model <- caret::train(
  SalePrice ~ .,  # log(SalePrice) ~ .
  data = train_dataset,
  method = "lm",
  trControl = ctrl
)

elastic_model <- caret::train(
  SalePrice ~ .,  # log(SalePrice) ~ .
  data = train_dataset,
  method = "glmnet",
  trControl = ctrl
)

boosted_model <- caret::train(
  SalePrice ~ .,  # log(SalePrice) ~ .
  data = train_dataset,
  method = "xgbLinear",
  trControl = ctrl
)



# validation metrics
print(bagged_model)
print(linear_model)
print(elastic_model)
print(boosted_model)


# fill Kaggle module for submission
sample_submission_path <- file.path("datasets", "house_prices_dataset", "sample_submission.csv")
sample_submission <- read.csv(sample_submission_path, stringsAsFactors = TRUE)

test_dataset <- dataset[(tl + 1):nrow(dataset),]
pred <- predict(linear_model, newdata = test_dataset)
sample_submission$SalePrice <- pred

write.csv(sample_submission, sample_submission_path, row.names=FALSE)
