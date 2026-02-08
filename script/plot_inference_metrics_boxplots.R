args <- commandArgs(trailingOnly = TRUE)

in_csv <- if (length(args) >= 1) args[[1]] else "inference_metrics.csv"
out_png <- if (length(args) >= 2) args[[2]] else "inference_metrics_boxplots.png"

if (!file.exists(in_csv)) {
  stop(paste("Input CSV not found:", in_csv))
}

if (!requireNamespace("ggplot2", quietly = TRUE) ||
    !requireNamespace("dplyr", quietly = TRUE) ||
    !requireNamespace("tidyr", quietly = TRUE)) {
  stop("Please install packages: ggplot2, dplyr, tidyr")
}

library(ggplot2)
library(dplyr)
library(tidyr)

df <- read.csv(in_csv, check.names = FALSE)

if (!("train_size" %in% colnames(df))) {
  df$train_size <- NA
}

df$checkpoint_label <- ifelse(
  !is.na(df$train_size) & df$train_size != "",
  as.character(df$train_size),
  as.character(seq_len(nrow(df)))
)

metrics_map <- list(
  acc = list(overall = "acc", suffix = "_acc"),
  precision = list(overall = "precision_macro", suffix = "_precision"),
  recall = list(overall = "recall_macro", suffix = "_recall"),
  f1 = list(overall = "f1_macro", suffix = "_f1")
)

class_long_list <- list()
overall_list <- list()

for (m in names(metrics_map)) {
  suffix <- metrics_map[[m]]$suffix
  overall_col <- metrics_map[[m]]$overall

  class_cols <- grep(paste0(suffix, "$"), colnames(df), value = TRUE)
  if (length(class_cols) == 0 || !(overall_col %in% colnames(df))) {
    next
  }

  class_long <- df %>%
    select(checkpoint_label, all_of(class_cols)) %>%
    pivot_longer(cols = all_of(class_cols), names_to = "class_metric", values_to = "value") %>%
    mutate(
      metric = m,
      class = sub(paste0(suffix, "$"), "", class_metric),
      value = as.numeric(value)
    ) %>%
    select(checkpoint_label, metric, class, value)

  overall <- df %>%
    select(checkpoint_label, all_of(overall_col)) %>%
    rename(overall = all_of(overall_col)) %>%
    mutate(
      metric = m,
      overall = as.numeric(overall)
    )

  class_long_list[[m]] <- class_long
  overall_list[[m]] <- overall
}

class_long_df <- bind_rows(class_long_list)
overall_df <- bind_rows(overall_list)

if (nrow(class_long_df) == 0) {
  stop("No per-class metrics found to plot.")
}

checkpoint_levels <- df %>%
  mutate(train_size_num = suppressWarnings(as.numeric(train_size))) %>%
  arrange(train_size_num, checkpoint_label) %>%
  pull(checkpoint_label) %>%
  unique()

class_long_df$checkpoint_label <- factor(class_long_df$checkpoint_label, levels = checkpoint_levels)
overall_df$checkpoint_label <- factor(overall_df$checkpoint_label, levels = checkpoint_levels)

p <- ggplot(class_long_df, aes(x = checkpoint_label, y = value, group = checkpoint_label)) +
  geom_boxplot(outlier.shape = NA, width = 0.6, alpha = 0.6) +
  geom_jitter(width = 0.15, size = 1.4, alpha = 0.8) +
  geom_errorbar(
    data = overall_df,
    aes(ymin = overall, ymax = overall),
    width = 0.5,
    color = "red",
    size = 0.6
  ) +
  facet_wrap(~metric, scales = "free_y", ncol = 2) +
  labs(
    x = "Train size",
    y = "Metric value",
    title = "Per-class Metrics with Overall Reference"
  ) +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

ggsave(out_png, plot = p, width = 10, height = 6, dpi = 200)
cat("Saved plot:", out_png, "\n")
