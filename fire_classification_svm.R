library(caret)
library(e1071)
library(OpenImageR)


# extracting features based on the gradients
extract_hog_features <- function(dataset) {
  feature_count <- 54
  col_names <- c(paste0("feature_", 1:feature_count), "label")
  features <- data.frame()
  
  for(i in 1:nrow(dataset)) {
    path <- dataset[i, "path"]
    label <- dataset[i, "label"]
    image <- readImage(path)
    sample_features <- HOG(image, cells = 3, orientations = 6)
    row <- c(sample_features, label)
    features <- rbind(features, row)
    print(path)
  }
  
  colnames(features) <- col_names
  features$label <- as.factor(features$label)
  return(features)
}

train_dataset_path <- file.path("datasets", "fire_dataset", "train_gray")
test_dataset_path <- file.path("datasets", "fire_dataset", "test_gray")

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

# extract features from the training set
train_dataset <- extract_hog_features(train_dataset)

# SVM
model <- svm(label ~ ., data = train_dataset, kernel = "polinomial", cost = 5)
summary(model)

# extract features from the test set
test_dataset <- extract_hog_features(test_dataset)

# evaluation
pred <- predict(model, train_dataset)
temp <- data.frame(pred = pred, true = train_dataset$label)
print(temp)
confusionMatrix(pred, train_dataset$labels)
