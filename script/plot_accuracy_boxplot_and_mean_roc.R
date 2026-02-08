args <- commandArgs(trailingOnly = TRUE)

roc_csv <- if (length(args) >= 1) args[[1]] else "roc_mean_curve.csv"
auc_csv <- if (length(args) >= 2) args[[2]] else "auc_per_label.csv"
out_roc_png <- if (length(args) >= 3) args[[3]] else "mean_roc_curve.png"
out_auc_png <- if (length(args) >= 4) args[[4]] else "auc_boxplot.png"

if (!file.exists(roc_csv)) {
  stop(paste("ROC file not found:", roc_csv))
}
if (!file.exists(auc_csv)) {
  stop(paste("AUC file not found:", auc_csv))
}

if (!requireNamespace("ggplot2", quietly = TRUE) || !requireNamespace("dplyr", quietly = TRUE)) {
  stop("Please install packages: ggplot2, dplyr")
}

library(ggplot2)
library(dplyr)

roc_df <- read.csv(roc_csv)
auc_df <- read.csv(auc_csv)

roc_df$file <- factor(roc_df$file, levels = unique(roc_df$file))
auc_df$file <- factor(auc_df$file, levels = unique(auc_df$file))

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr_mean, color = file, group = file)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate",
    color = "File",
    title = "Macro-average ROC (7 labels)"
  ) +
  theme_minimal()

ggsave(out_roc_png, plot = p_roc, width = 7, height = 5, dpi = 200)

p_auc <- ggplot(auc_df, aes(x = file, y = auc, fill = file)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 1.5, alpha = 0.8) +
  labs(
    x = "File",
    y = "AUC",
    title = "AUC per Label (Boxplot)"
  ) +
  theme_minimal() +
  theme(legend.position = "none")

ggsave(out_auc_png, plot = p_auc, width = 7, height = 5, dpi = 200)

cat("Saved ROC plot:", out_roc_png, "\n")
cat("Saved AUC boxplot:", out_auc_png, "\n")
