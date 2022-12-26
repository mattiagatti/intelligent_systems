fire_images_path <- file.path("fire_dataset", "fire_images")
non_fire_images_path <- file.path("fire_dataset", "non_fire_images")

train_dataset_path <- file.path("datasets", "fire_dataset", "train")
test_dataset_path <- file.path("datasets", "fire_dataset", "test")

fire_images <- list.files(path = fire_images_path, full.names = TRUE)
non_fire_images <- list.files(path = non_fire_images_path, full.names = TRUE)
positives_df <- data.frame(path = fire_images, label = "fire")
negatives_df <- data.frame(path = non_fire_images, label = "non_fire")

# split positives
set.seed(42)
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

# create directories
dir.create(file.path(train_dataset_path, "fire"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(train_dataset_path, "non_fire"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(test_dataset_path, "fire"), recursive = TRUE, showWarnings = FALSE)
dir.create(file.path(test_dataset_path, "non_fire"), recursive = TRUE, showWarnings = FALSE)

# copy train samples to the respective folder
for(i in 1:nrow(train_dataset_df)) {
  path <- train_dataset_df[i,"path"]
  label <- train_dataset_df[i,"label"]
  new_path <- file.path(train_dataset_path, label, basename(path))
  file.copy(from = path, to = new_path,
            overwrite = TRUE, recursive = FALSE, 
            copy.mode = TRUE)
  print(new_path)
}

# copy test samples to the respective folder
for(i in 1:nrow(test_dataset_df)) {
  path <- test_dataset_df[i,"path"]
  label <- test_dataset_df[i,"label"]
  new_path <- file.path(test_dataset_path, label, basename(path))
  file.copy(from = path, to = new_path,
            overwrite = TRUE, recursive = FALSE, 
            copy.mode = TRUE)
  print(new_path)
}