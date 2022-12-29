library(tensorflow)
library(keras)
library(reticulate)
library(e1071)
library(class)
library(caret)

img_size <- c(256L, 256L)
img_shape <- c(img_size, 3L)

cnn_feature_extractor <- application_efficientnet_b0(
  include_top = FALSE,
  weights = "imagenet",
  pooling = "max",
  input_shape = img_shape
)

feature_count <- as.integer(cnn_feature_extractor$output_shape[2])
col_names <- c(paste0("feature_", 1:feature_count), "label")

extract_features <- function(dataset) {
  features <- data.frame()
  
  for(i in 1:nrow(dataset)) {
    path <<- dataset[i, "path"]
    label <- dataset[i, "label"]
    image <- keras$utils$load_img(path, target_size = img_size)
    input_arr <<- keras$utils$img_to_array(image)
    py_run_string("import numpy as np")
    py_run_string("import tensorflow as tf")
    py_run_string("input_arr = np.array([r.input_arr])")
    py_run_string("sample_features = r.cnn_feature_extractor(input_arr).numpy().squeeze()")
    sample_features <- py$sample_features
    row <- c(sample_features, label)
    features <- rbind(features, row)
    print(paste(i, nrow(dataset), sep = " / "))
  }
  colnames(features) <- col_names
  features$label <- as.factor(features$label)
  return(features)
}

dataset_path <- file.path("datasets", "fire_dataset")
train_dataset_path <- file.path(dataset_path, "train")
test_dataset_path <- file.path(dataset_path, "test")

train_images <- list.files(path = train_dataset_path, full.names = TRUE, recursive = TRUE)
test_images <- list.files(path = test_dataset_path, full.names = TRUE, recursive = TRUE)

train_labels <- basename(dirname(train_images))
train_labels[train_labels == "non_fire"] <- 0
train_labels[train_labels == "fire"] <- 1
test_labels <- basename(dirname(test_images))
test_labels[test_labels == "non_fire"] <- 0
test_labels[test_labels == "fire"] <- 1

train_dataset <- data.frame(path = train_images, label = train_labels)
test_dataset <- data.frame(path = test_images, label = test_labels)

# shuffle
set.seed(42)
train_dataset <- train_dataset[sample(1:nrow(train_dataset)), ]


train_df_path = file.path(dataset_path, "train_dataset.Rda")
test_df_path = file.path(dataset_path, "test_dataset.Rda")

# extract features for each image in the train set
if(!file.exists(train_df_path)) {
  train_dataset <- extract_features(train_dataset)
  saveRDS(train_dataset, file = train_df_path)
} else {
  train_dataset <- readRDS(train_df_path)
}

# extract features for each image in the test set
if(!file.exists(test_df_path)) {
  test_dataset <- extract_features(test_dataset)
  saveRDS(test_dataset, file=file.path(test_df_path))
} else {
  test_dataset <- readRDS(test_df_path)
}

# K-NN
perform_knn <- function(test, k) {
  cl <- train_dataset[, c(ncol(train_dataset))]  # train label
  train <- train_dataset[, c(1:ncol(train_dataset) - 1)]  # train features
  cl_test <- test$label  # test labels
  test <- test[, c(1:ncol(test) - 1)]  # test features
  
  knn <- knn(train = train, test = test, cl = cl, k = k)
  confusionMatrix(table(knn, cl_test))
}

# found best k with cross validation
best_k <- 5
perform_knn(test_dataset, best_k)

# final evaluation
final_test_dataset_path <- file.path("datasets", "forest_fire_dataset")
final_test_images <- list.files(path = final_test_dataset_path, full.names = TRUE, recursive = TRUE)
final_test_labels <- basename(dirname(final_test_images))
final_test_labels[final_test_labels == "non_fire"] <- 0
final_test_labels[final_test_labels == "fire"] <- 1
final_test_dataset <- data.frame(path = final_test_images, label = final_test_labels)
final_test_df_path <- file.path(final_test_dataset_path, "test_dataset.Rda")

if(!file.exists(final_test_df_path)) {
  final_test_dataset <- extract_features(final_test_dataset)
  saveRDS(final_test_dataset, file=file.path(final_test_df_path))
} else {
  final_test_dataset <- readRDS(final_test_df_path)
}

perform_knn(final_test_dataset, best_k)


