library("optparse")

option_list = list(
  make_option("--TP", type = "integer", default = NULL, 
              help = "Number of true positives in the confusion matrix", metavar = "integer"),
  make_option("--FP", type = "integer", default = NULL, 
              help = "Number of false positives in the confusion matrix", metavar = "integer"),
  make_option("--FN", type = "integer", default = NULL, 
              help = "Number of false negatives in the confusion matrix", metavar = "integer"),
  make_option("--TN", type = "integer", default = NULL, 
              help = "Number of false negatives in the confusion matrix", metavar = "integer")
);

opt_parser = OptionParser(option_list=option_list);
opt = parse_args(opt_parser);
if (length(opt) != 5){
  print_help(opt_parser)
  stop("At least four argument must be supplied (TP, FP, FN, TN)", call. = FALSE)
}

TP <- opt$TP
FP <- opt$FP
FN <- opt$FN
TN <- opt$TN
N <- TP + FP + FN + TN
Po <- (TP + TN) / N  # observed
Pe <- (TP + FN) / N * (TP + FP) / N + (FP + TN) / N * (FN + TN) / N  # expected
Kappa <- (Po - Pe) / (1 - Pe)
print(Kappa)
