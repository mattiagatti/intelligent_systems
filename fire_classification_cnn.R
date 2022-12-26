library(tensorflow)
tf$config$list_physical_devices("GPU")
exit
library(keras)

seed <- 42
img_height <- 600
img_width <- 600
batch_size <- 2
epochs <- 1L
initial_lr <- 0.001
weight_decay <- 0.0001
train_dataset_path <- file.path("datasets", "fire_dataset", "train")
test_dataset_path <- file.path("datasets", "fire_dataset", "test")


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
model$add(keras$layers$Dense(units = 120, activation = "relu"))
model$add(keras$layers$Dense(units = 120, activation = "relu"))
model$add(keras$layers$Dense(units = 1, activation = "sigmoid"))

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

history = model$fit(train_dataset, validation_data = val_dataset, epochs = epochs)
results = model$evaluate(test_images, verbose=0)