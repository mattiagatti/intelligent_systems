library(e1071)
library(imager)
library(OpenImageR)

fire_images <- list.files(path="./fire_dataset/1", full.names=TRUE)
non_fire_images <- list.files(path="./fire_dataset/0", full.names=TRUE)

positives <- data.frame(path = fire_images, label = 1)
negatives <- data.frame(path = non_fire_images, label = 0)
dataset <- rbind(positives, negatives)

# image = load.image(positives$path[1])
# image = resize(image, size_x = 600, size_y = 600)
# print(image)
# image = grayscale(z)
# print(positives$path[1])
# image = readImage(positives$path[1])
# res = HOG(image, cells = 3, orientations = 6)
# print(res)
