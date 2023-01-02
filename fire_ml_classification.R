library(caret)
library(keras)
library(tensorflow)
library(reticulate)
library(ggplot2)

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

fix_datatypes <- function(df) {
  cols.num <- 1:(ncol(df) - 1)
  df[cols.num] <- lapply(df[cols.num], as.numeric)
  df$label <-  as.factor(df$label)
  return(df)
}

print_metrics <- function(pred, target) {
  cm <- confusionMatrix(pred, target, positive = "1", mode = "prec_recall")
  overall <- round(cm$overall, digits = 4)
  byclass <- round(cm$byClass, digits = 4)
  print(cm)
  # Accuracy, Kappa, Precision, Recall, F1
  print(paste(overall[1], overall[2], byclass[5], byclass[6], byclass[7], sep = " & "))
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
train_dataset <- fix_datatypes(train_dataset)

# extract features for each image in the test set
if(!file.exists(test_df_path)) {
  test_dataset <- extract_features(test_dataset)
  saveRDS(test_dataset, file=file.path(test_df_path))
} else {
  test_dataset <- readRDS(test_df_path)
}
test_dataset <- fix_datatypes(test_dataset)

# CV
ctrl <- caret::trainControl(method="cv", number = 10)

# K-NN
knnFit <- caret::train(label ~ ., data = train_dataset, method = "knn", trControl = ctrl, preProcess = c("center","scale"), tuneLength = 20)

# plot best K graph
ggplot(data = knnFit$results, aes(k, Accuracy)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = knnFit$results$k) +
  scale_y_continuous(breaks = seq(0, 1, 0.005))

# test the knn fit
knnPredict <- predict(knnFit, newdata = test_dataset)
print_metrics(knnPredict, test_dataset$label)

# SVM
svmFit <- caret::train(label ~ ., data = train_dataset, method = "svmLinear", trControl = ctrl,  preProcess = c("center","scale"), tuneGrid = expand.grid(C = c(0.01, seq(0.05, 1, length = 20))))

# plot best C graph
ggplot(data = svmFit$results, aes(C, Accuracy)) +
  geom_line() +
  geom_point() +
  scale_x_continuous(breaks = seq(0, 1, 0.05))

# test the svm fit
svmPredict <- predict(svmFit, newdata = test_dataset)
print_metrics(svmPredict, test_dataset$label)


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

final_test_dataset <- fix_datatypes(final_test_dataset)

# final eval knn
knnPredict <- predict(knnFit, newdata = final_test_dataset)
print_metrics(knnPredict, final_test_dataset$label)

# final eval svm
svmPredict <- predict(svmFit, newdata = final_test_dataset)
print_metrics(svmPredict, final_test_dataset$label)


