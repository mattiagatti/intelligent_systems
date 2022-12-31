library(tensorflow)
library(keras)
library(stringr)
library(tidyverse)
library(ggplot2)
library(cvms)
library(tibble)

seed <- 42
img_size <- c(256L, 256L)
img_shape <- c(img_size, 3L)
batch_size <- 32L
epochs <- 10L
initial_lr <- 0.0001  # low lr for fine tuning
weight_decay <- 0.0001
model_name <- "efficient_net_b0"
train_dataset_path <- file.path("datasets", "fire_dataset", "train")
test_dataset_path <- file.path("datasets", "fire_dataset", "test")
big_test_dataset_path <- file.path("datasets", "forest_fire_dataset", "train")
checkpoint_dir <- file.path("training", model_name)
checkpoint_path <- file.path(checkpoint_dir, "cp-list{epoch:04d}.ckpt")
train <- FALSE  # choose if train or evaluate only


# defining the feature extractor (transfer learning)
if(model_name == "efficient_net_b0") {
  efficient_net <- application_efficientnet_b0(
    include_top = FALSE,
    weights = "imagenet",
    pooling = "max",
    input_shape = img_shape
  )
} else if(model_name == "efficient_net_b3") {
  efficient_net <- application_efficientnet_b3(
    include_top = FALSE,
    weights = "imagenet",
    pooling = "max",
    input_shape = img_shape
  )
} else if(model_name == "efficient_net_b7") {
  efficient_net <- application_efficientnet_b7(
    include_top = FALSE,
    weights = "imagenet",
    pooling = "max",
    input_shape = img_shape
  )
} else if(model_name == "resnet_50") {
    efficient_net <- application_resnet50(
      include_top = FALSE,
      weights = "imagenet",
      pooling = "max",
      input_shape = img_shape
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
  image_size = img_size,
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
  image_size = img_size,
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
  image_size = img_size,
)

big_test_dataset <- image_dataset_from_directory(
  big_test_dataset_path,
  labels = "inferred",
  label_mode = "int",
  color_mode = "rgb",
  batch_size = batch_size,
  image_size = img_size,
)

# cosine annealing scheduler
lr_cosine_decay <- function(epoch, lr) {
  return (0.5 * initial_lr * (1 + cos(epoch / epochs * pi)))
}

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

if(train == TRUE) {
  history = model$fit(
    train_dataset,
    epochs = epochs,
    validation_data = val_dataset,
    callbacks = list(checkpoint_callback, keras$callbacks$LearningRateScheduler(lr_cosine_decay, verbose=1))
  )
  
  # save model history
  history_df <- as.data.frame(history$history)
  history_df <- history_df[1:(length(history_df)-epochs)]
  saveRDS(history_df, file=file.path(checkpoint_dir, "history.Rda"))
}

# load model history to find best weights
history_df <- readRDS(file=file.path(checkpoint_dir, "history.Rda"))
best_epoch <- which.max(history_df$val_accuracy)
best_epoch <- str_pad(best_epoch, 4, pad = "0")

# Loads the weights and test the model
best_checkpoint_path <- file.path(checkpoint_dir, paste0("cp-list", best_epoch , ".ckpt"))
load_model_weights_tf(model, best_checkpoint_path)
test_metrics <- model %>% evaluate(test_dataset, verbose = 1)
big_test_metrics <- model %>% evaluate(big_test_dataset, verbose = 1)

# plot cm from metrics
cm <- c(big_test_metrics[8], big_test_metrics[6], big_test_metrics[5], big_test_metrics[7])
cm <- matrix(cm, ncol=2, byrow=TRUE)
colnames(cm) <- c('positive','negative')
rownames(cm) <- c('positive','negative')
cm <- as.table(cm)
cm <- as_tibble(cm, .name_repair = ~c("target", "prediction", "n"))
plot_confusion_matrix(cm, 
                      target_col = "target", 
                      prediction_col = "prediction",
                      counts_col = "n")

# plot loss vs accuracy
lines <- history_df[,c("loss","val_accuracy")]
lines$epoch <- c(1:nrow(lines))
print(lines)

lines <- lines %>%
  select(epoch, loss, val_accuracy) %>%
  gather(key = "variable", value = "value", -epoch)

ggplot(lines, aes(x = epoch, y = value)) + 
  geom_line(aes(color = variable)) + 
  scale_color_manual(values = c("red", "blue")) +
  scale_x_continuous(breaks = 1:10) +
  scale_y_continuous(breaks = (0:20)/20)
