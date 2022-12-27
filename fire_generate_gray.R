library(imager)

train_dataset_path <- file.path("datasets", "fire_dataset", "train")
test_dataset_path <- file.path("datasets", "fire_dataset", "test")

train_images <- list.files(path = train_dataset_path, full.names = TRUE, recursive = TRUE)
test_images <- list.files(path = test_dataset_path, full.names = TRUE, recursive = TRUE)

for (image_path in c(train_images, test_images)) {
  print(image_path)
  try({
    image <- load.image(image_path)
    image <- resize(image, size_x = 600, size_y = 600)
    image <- grayscale(image)
    save_path <- sub("train", "train_gray", image_path)
    save_path <- sub("test", "test_gray", save_path)
    dir.create(dirname(save_path), recursive = TRUE, showWarnings = FALSE)
    save.image(image, save_path, quality = 1)
    }
  )
}