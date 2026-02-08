args <- commandArgs(trailingOnly = TRUE)

in_csv <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/inference_metrics.csv"
out_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/inference_metrics_plots"

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
metric_labels <- c(
  acc = "Accuracy",
  precision = "Precision",
  recall = "Recall",
  f1 = "F1"
)

class_long_list <- list()

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

  class_long_list[[m]] <- class_long
}

class_long_df <- bind_rows(class_long_list)

if (nrow(class_long_df) == 0) {
  stop("No per-class metrics found to plot.")
}

checkpoint_levels <- df %>%
  mutate(train_size_num = suppressWarnings(as.numeric(train_size))) %>%
  arrange(train_size_num, checkpoint_label) %>%
  pull(checkpoint_label) %>%
  unique()

class_long_df$checkpoint_label <- factor(class_long_df$checkpoint_label, levels = checkpoint_levels)
checkpoint_sizes <- df %>%
  mutate(train_size_num = suppressWarnings(as.numeric(train_size))) %>%
  select(checkpoint_label, train_size_num)
size_lookup <- checkpoint_sizes$train_size_num[match(checkpoint_levels, checkpoint_sizes$checkpoint_label)]
max_size <- max(size_lookup, na.rm = TRUE)
percent_labels <- if (is.finite(max_size)) {
  pct <- round((size_lookup / max_size) * 100 / 10) * 10
  pct[is.na(pct)] <- ""
  paste0(pct, "%")
} else {
  rep("", length(checkpoint_levels))
}

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

for (m in unique(class_long_df$metric)) {
  df_m <- class_long_df %>% filter(metric == m)
  if (nrow(df_m) == 0) {
    next
  }

  p <- ggplot(df_m, aes(x = checkpoint_label, y = value, group = checkpoint_label)) +
    geom_boxplot(outlier.shape = NA, width = 0.6, fill = "#6baed6", color = "#2166ac") +
    geom_jitter(width = 0.15, size = 1.4, alpha = 0.8, color = "#1b4f72") +
    labs(
      x = "Train size",
      y = metric_labels[[m]]
    ) +
    scale_x_discrete(labels = percent_labels) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

  out_png <- file.path(out_dir, paste0("boxplot_", m, ".png"))
  ggsave(out_png, plot = p, width = 8, height = 5, dpi = 200)
  cat("Saved plot:", out_png, "\n")
}

curve_map <- list(
  acc = "acc",
  auc = "auc_macro",
  f1 = "f1_macro"
)
curve_labels <- c(
  acc = "Accuracy",
  auc = "AUC",
  f1 = "F1"
)

curve_df <- lapply(names(curve_map), function(k) {
  col_name <- curve_map[[k]]
  if (!(col_name %in% colnames(df))) {
    return(NULL)
  }
  data.frame(
    checkpoint_label = df$checkpoint_label,
    metric = k,
    value = as.numeric(df[[col_name]])
  )
}) %>% bind_rows()

if (nrow(curve_df) > 0) {
  curve_df$checkpoint_label <- factor(curve_df$checkpoint_label, levels = checkpoint_levels)
  curve_df$metric <- factor(curve_df$metric, levels = names(curve_map))

  p_curve <- ggplot(curve_df, aes(x = checkpoint_label, y = value, color = metric, group = metric)) +
    geom_line(size = 0.8) +
    geom_point(size = 1.8) +
    scale_color_discrete(labels = curve_labels) +
    scale_x_discrete(labels = percent_labels) +
    labs(
      x = "Train size",
      y = "Metric value",
      color = ""
    ) +
    theme_minimal() +
    theme(
      axis.text.x = element_text(angle = 0, hjust = 0.5),
      legend.title = element_blank()
    )

  out_curve <- file.path(out_dir, "curve_loss_acc_auc_f1.png")
  ggsave(out_curve, plot = p_curve, width = 8, height = 5, dpi = 200)
  cat("Saved plot:", out_curve, "\n")
}

if ("loss" %in% colnames(df)) {
  loss_df <- data.frame(
    checkpoint_label = factor(df$checkpoint_label, levels = checkpoint_levels),
    loss = as.numeric(df$loss)
  )

  p_loss <- ggplot(loss_df, aes(x = checkpoint_label, y = loss, group = 1)) +
    geom_line(size = 0.6, color = "#2c7fb8") +
    geom_point(size = 2, color = "#2c7fb8") +
    scale_x_discrete(labels = percent_labels) +
    labs(
      x = "Train size",
      y = "Test Loss"
    ) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 0, hjust = 0.5))

  out_loss <- file.path(out_dir, "curve_test_loss.png")
  ggsave(out_loss, plot = p_loss, width = 8, height = 5, dpi = 200)
  cat("Saved plot:", out_loss, "\n")
}
