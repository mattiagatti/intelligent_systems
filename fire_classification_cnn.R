library(tensorflow)
library(keras)

seed <- 42
img_height <- 600
img_width <- 600
batch_size <- 2
epochs <- 1L
initial_lr <- 0.001
weight_decay <- 0.0001
train_dataset_path <- "./datasets/fire_dataset/train"
test_dataset_path <- "./datasets/fire_dataset/test"


generate_splits <- function() {
  # load dataset
  fire_images <- list.files(path="./fire_dataset/fire_images", full.names=TRUE)
  non_fire_images <- list.files(path="./fire_dataset/non_fire_images", full.names=TRUE)
  positives_df <- data.frame(path = fire_images, label = "fire")
  negatives_df <- data.frame(path = non_fire_images, label = "non_fire")
  
  # split positives
  set.seed(seed)
  sample <- sample.int(n = nrow(positives_df), size = (nrow(positives_df) - 50), replace = F)
  train_positives_df <- positives_df[sample, ]
  test_positives_df  <- positives_df[-sample, ]
  
  # split negatives
  sample <- sample.int(n = nrow(negatives_df), size = (nrow(negatives_df) - 50), replace = F)
  train_negatives_df <- negatives_df[sample, ]
  test_negatives_df  <- negatives_df[-sample, ]
  
  # merge positives and negatives
  train_dataset_df <- rbind(train_positives_df, train_negatives_df)
  test_dataset_df <- rbind(test_positives_df, test_negatives_df)
  
  for(i in 1:nrow(train_dataset_df)) {
    path <- train_dataset_df[i,"path"]
    label <- train_dataset_df[i,"label"]
    new_path <- paste(train_dataset_path, label, basename(path), sep = "/")
    dir.create(new_path)
    file.copy(from = path, to = new_path, 
              overwrite = TRUE, recursive = FALSE, 
              copy.mode = TRUE)
  }
}

# defining the feature extractor with transfer learning
efficient_net <- application_efficientnet_b7(
  include_top = FALSE,
  weights = "imagenet",
  pooling = "max"
)


# to print the new learning rate
get_lr_metric <- function(optimizer) {
  lr <- function(y_true, y_pred) {
    return (optimizer$lr)
  }
  return(lr)
}


# adding to the feature extractor the head to perform binary classification
model = keras$Sequential()
model$add(efficient_net)
model$add(keras$layers$Dense(units = 120, activation = 'relu'))
model$add(keras$layers$Dense(units = 120, activation = 'relu'))
model$add(keras$layers$Dense(units = 1, activation = 'sigmoid'))

generate_splits()
exit()

# define tensorflow train and test datasets
train_dataset <- image_dataset_from_directory(
  train_dataset_path,
  labels = "inferred",
  label_mode = "int",
  color_mode = "rgb",
  batch_size = batch_size,
  image_size = c(img_height, img_width),
  shuffle = TRUE,
  seed = seed,
  validation_split = 0.1,
  subset = "training"
)

val_dataset <- image_dataset_from_directory(
  train_dataset_path,
  labels = "inferred",
  label_mode = "int",
  color_mode = "rgb",
  batch_size = batch_size,
  image_size = c(img_height, img_width),
  shuffle = TRUE,
  seed = seed,
  validation_split = 0.1,
  subset = "validation"
)

test_dataset <- image_dataset_from_directory(
  test_dataset_path,
  labels = "inferred",
  label_mode = "int",
  color_mode = "rgb",
  batch_size = batch_size,
  image_size = c(img_height, img_width),
  shuffle = TRUE,
  seed = seed
)

# the number of times the lr is decreased is the number of batch for each epoch
# per the number of epochs the learning rate will reach the limit of 0 at the last batch
cardinality <- train_dataset$cardinality()$numpy()
decay_steps <- epochs * as.integer(cardinality)

lr_schedule = keras$optimizers$schedules$CosineDecay(
  initial_learning_rate = initial_lr,
  decay_steps = decay_steps
)
optimizer = keras$optimizers$Adam(learning_rate = lr_schedule, weight_decay = weight_decay)
new_lr = get_lr_metric(optimizer)

model %>% compile(
  optimizer = optimizer,
  loss = "binary_crossentropy",
  metrics = c("accuracy", keras$metrics$Precision(), keras$metrics$Recall(), new_lr)
)

model$fit(train_dataset, validation_data = val_dataset, epochs = epochs)