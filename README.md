Datasets:
- FIRE Dataset: https://www.kaggle.com/datasets/phylake1337/fire-dataset
- Forest Fire Dataset: https://www.kaggle.com/datasets/mohnishsaiprasad/forest-fire-images/code
- Landscape Dataset: https://www.kaggle.com/datasets/arnaud58/landscape-pictures

The first two datasets have to be copied inside the datasets folder (adjust the paths if you choose different locations). The split script is used to split the first dataset between train set and test set (to compare the models without validation set bias). The second dataset (train and test merged in a single test split) is used as a big test set (4050 images after removing corrupted files) to evaluate model performance outside the starting dataset. The last dataset dataset can be useful to balance the number of non_fire samples of the first dataset. 

To install TensorFlow for R:
1. Install RTools: https://cran.r-project.org/bin/windows/Rtools/
2. Install Anaconda: https://www.anaconda.com/products/distribution
3. install.packages(c("keras", "tensorflow"))
4. install.packages("devtools")
5. library(devtools)
6. devtools::install_github("rstudio/keras", dependencies = TRUE)
7. devtools::install_github("rstudio/tensorflow", dependencies = TRUE)
8. library(keras)
9. library(tensorflow)
10. install_keras()
11. install_tensorflow()
12. Install CUDA: https://developer.nvidia.com/cuda-11.2.2-download-archive
13. Install cuDNN: https://developer.nvidia.com/rdp/cudnn-archive