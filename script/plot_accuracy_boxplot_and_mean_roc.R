args <- commandArgs(trailingOnly = TRUE)

roc_csv <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/roc_mean_curve.csv"
auc_csv <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/auc_per_label.csv"
out_roc_png <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/label_roc_auc/mean_roc_curve.png"
out_auc_png <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/label_roc_auc/auc_boxplot.png"

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

auc_mean_df <- auc_df %>%
  group_by(file) %>%
  summarize(auc_mean = mean(auc, na.rm = TRUE), .groups = "drop") %>%
  mutate(
    label = gsub("^calibration_long-", "predict-roc-", file),
    label = paste0(label, " (AUC=", sprintf("%.3f", auc_mean), ")")
  )

roc_labels <- setNames(auc_mean_df$label, auc_mean_df$file)

p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr_mean, color = file, group = file)) +
  geom_line(size = 1) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "gray50") +
  labs(
    x = "False Positive Rate",
    y = "True Positive Rate",
    color = "",
    title = "Macro-average ROC (7 labels)"
  ) +
  scale_color_discrete(labels = roc_labels) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggsave(out_roc_png, plot = p_roc, width = 7, height = 5, dpi = 200)

p_auc <- ggplot(auc_df, aes(x = file, y = auc, fill = file)) +
  geom_boxplot(outlier.shape = NA, alpha = 0.7) +
  geom_jitter(width = 0.15, size = 1.2, alpha = 0.5, color = "gray30") +
  labs(
    x = "",
    y = "AUC",
    title = "AUC per Label"
  ) +
  scale_x_discrete(labels = function(x) gsub("^(calibration_long-|predict-roc-)", "", x)) +
  theme_minimal() +
  theme(
    legend.position = "none",
    plot.title = element_text(hjust = 0.5)
  )

ggsave(out_auc_png, plot = p_auc, width = 7, height = 5, dpi = 200)

cat("Saved ROC plot:", out_roc_png, "\n")
cat("Saved AUC boxplot:", out_auc_png, "\n")
