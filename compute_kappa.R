TP <- 1870
FP <- 70
FN <- 130
TN <- 1930
N <- TP + FP + FN + TN
Po <- (TP + TN) / N
Pe <- (TP + FN) / N * (TP + FP) / N * (FP + TN) / N * (FN + TN) / N
Kappa <- (Po - Pe) / (1 - Pe)
Kappa