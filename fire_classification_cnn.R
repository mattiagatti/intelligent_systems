library(tensorflow)
library(keras)

seed <- 42
img_height <- 100
img_width <- 100
batch_size <- 2
epochs <- 10L
initial_lr <- 0.0001
weight_decay <- 0.0001
model_size <- "b0"
train_dataset_path <- file.path("datasets", "fire_dataset", "train")
test_dataset_path <- file.path("datasets", "fire_dataset", "test")
checkpoint_path <- file.path("training", paste0("efficient_net_", model_size), "cp-list{epoch:04d}.ckpt")
best_checkpoint_path <- file.path("training", paste0("efficient_net_", model_size), "cp-list0001.ckpt")


# defining the feature extractor with transfer learning
if(model_size == "b0") {
  efficient_net <- application_efficientnet_b0(
    include_top = FALSE,
    weights = "imagenet",
    pooling = "max"
  )
} else if(model_size == "b7") {
  efficient_net <- application_efficientnet_b7(
    include_top = FALSE,
    weights = "imagenet",
    pooling = "max"
  )
}

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

# halven the lr after each epoch
lr_step_decay <- function(epoch, lr) {
  drop_rate <- 0.5
  epochs_drop <- 1.0
  return (initial_lr * drop_rate^(floor(epoch / epochs_drop)))
}

optimizer = keras$optimizers$Adam(weight_decay = weight_decay)

# Create a callback that saves the model's weights
checkpoint_callback = callback_model_checkpoint(
  checkpoint_path,
  save_weights_only = TRUE,
  # save_freq = num_train_batches,
  verbose = 1
)

metrics = c(
  "accuracy",
  keras$metrics$Precision(),
  keras$metrics$Recall(),
  keras$metrics$FalseNegatives(),
  keras$metrics$FalsePositives(),
  keras$metrics$TrueNegatives(),
  keras$metrics$TruePositives()
)

model %>% compile(
  optimizer = optimizer,
  loss = "binary_crossentropy",
  metrics = metrics
)

history = model$fit(
  train_dataset,
  epochs = epochs,
  validation_data = val_dataset,
  callbacks = list(checkpoint_callback, keras$callbacks$LearningRateScheduler(lr_step_decay, verbose=1))
)

# Loads the weights and test the model
load_model_weights_tf(model, best_checkpoint_path)
restored_model <- model %>% evaluate(test_dataset, verbose = 2)
