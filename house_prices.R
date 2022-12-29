train_dataset_path <- file.path("datasets", "house_prices_dataset", "train.csv")
test_dataset_path <- file.path("datasets", "house_prices_dataset", "test.csv")

train_dataset <- read.csv(train_dataset_path)
test_dataset <- read.csv(test_dataset_path)

colnames(train_dataset)