TP <- 1882
FP <- 52
FN <- 105
TN <- 1935
N <- TP + FP + FN + TN
Po <- (TP + TN) / N
Pe <- (TP + FN) / N * (TP + FP) / N + (FP + TN) / N * (FN + TN) / N
Kappa <- (Po - Pe) / (1 - Pe)
Kappa
