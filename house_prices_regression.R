train_dataset_path <- file.path("datasets", "house_prices_dataset", "train.csv")
test_dataset_path <- file.path("datasets", "house_prices_dataset", "test.csv")

train_dataset <- read.csv(train_dataset_path)
test_dataset <- read.csv(test_dataset_path)

sample_count <- nrow(train_dataset)

# printing descriptive statistics for each feature
summary(train_dataset)

# drop features with more than 1/3 NA values
train_dataset <- train_dataset[, colSums(is.na(train_dataset)) <= sample_count / 3]