library(tensorflow)
library(keras)
library(reticulate)
library(e1071)


efficient_net <- application_efficientnet_b0(
  include_top = FALSE,
  weights = "imagenet",
  pooling = "max",
  input_shape = c(img_height, img_width)
)

model = keras$Sequential()
model$add(efficient_net)
model$add(keras$layers$Dense(units = 120, activation = "relu"))
model$add(keras$layers$Dense(units = 120, activation = "relu"))
model$add(keras$layers$Dense(units = 1, activation = "sigmoid"))

checkpoint_path <- file.path("training", "efficient_net_b0", "cp-list0006.ckpt")
load_model_weights_tf(model, checkpoint_path)

# removing last two layers
pop_layer(model)
pop_layer(model)

feature_count <- 120
col_names <- c(paste0("feature_", 1:feature_count), "label")

extract_features <- function(dataset) {
  features <- data.frame()
  
  for(i in 1:nrow(dataset)) {
    path <- dataset[i, "path"]
    label <- dataset[i, "label"]
    image <- keras$utils$load_img(path)
    input_arr <- keras$utils$img_to_array(image)
    py_run_string("import tensorflow as tf")
    py_run_string("input_arr = np.array([r.input_arr])")
    py_run_string("sample_features = r.model(input_arr).numpy().squeeze()")
    sample_features <- py$sample_features
    row <- c(sample_features, label)
    features <- rbind(features, row)
    print(paste(i, nrow(dataset), sep = " / "))
  }
  colnames(features) <- col_names
  train_features$label <- as.factor(train_features$label)
  return(features)
}

train_dataset_path <- file.path("datasets", "fire_dataset", "train")
test_dataset_path <- file.path("datasets", "fire_dataset", "test")

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

# extract features for each image
train_dataset <- extract_features(train_dataset)
saveRDS(train_dataset, file=file.path("datasets", "fire_dataset", "train_dataset.Rda"))

# SVM
model <- svm(label ~ ., data = train_dataset, kernel = "radial")
summary(model)

# test model
test_dataset <- extract_features(test_dataset)
saveRDS(test_dataset, file=file.path("datasets", "fire_dataset", "test_dataset.Rda"))
pred <- predict(model, test_dataset)
