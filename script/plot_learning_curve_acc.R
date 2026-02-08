args <- commandArgs(trailingOnly = TRUE)

root_dir <- if (length(args) >= 1) args[[1]] else "."
out_png <- if (length(args) >= 2) args[[2]] else "learning_curve_val_acc.png"
out_csv <- if (length(args) >= 3) args[[3]] else "learning_curve_val_acc.csv"
span <- if (length(args) >= 4) as.numeric(args[[4]]) else 0.6

files <- list.files(root_dir, pattern = "eval_log\\.csv$", recursive = TRUE, full.names = TRUE)
if (length(files) == 0) {
  stop(paste("No eval_log.csv found under:", root_dir))
}

read_one <- function(path) {
  df <- read.csv(path)
  size_match <- regexpr("size_([0-9]+)_seed", path, perl = TRUE)
  train_size <- if (size_match[1] > 0) {
    as.integer(sub(".*size_([0-9]+)_seed.*", "\\1", path))
  } else {
    NA_integer_
  }
  df$train_size <- train_size
  df$train_group <- if (!is.na(train_size)) {
    as.character(train_size)
  } else {
    basename(dirname(path))
  }
  df$source <- basename(dirname(path))
  df
}

df_all <- do.call(rbind, lapply(files, read_one))
if (any(is.na(df_all$train_size))) {
  warning("Some paths did not match size_XXX_seed_YYY; use parent folder as group label.")
}

if (!requireNamespace("dplyr", quietly = TRUE) || !requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Please install packages: dplyr, ggplot2")
}

library(dplyr)
library(ggplot2)

# 每个 epoch 取一条记录（取 global_step 最大的那条）
df_epoch <- df_all %>%
  group_by(train_group, epoch) %>%
  slice_max(order_by = global_step, n = 1, with_ties = FALSE) %>%
  ungroup()

# 只保留所有 train_size 都具备的 epoch（保证对齐）
epoch_lists <- df_epoch %>%
  group_by(train_group) %>%
  summarise(epochs = list(sort(unique(epoch))), .groups = "drop") %>%
  pull(epochs)

common_epochs <- Reduce(intersect, epoch_lists)
if (length(common_epochs) > 0) {
  df_epoch <- df_epoch %>% filter(epoch %in% common_epochs)
}

if (all(!is.na(df_epoch$train_size))) {
  df_epoch$train_group <- factor(df_epoch$train_group, levels = as.character(sort(unique(df_epoch$train_size))))
} else {
  df_epoch$train_group <- factor(df_epoch$train_group, levels = unique(df_epoch$train_group))
}

write.csv(df_epoch, out_csv, row.names = FALSE)

p <- ggplot(df_epoch, aes(x = epoch, y = val_acc, color = train_group, group = train_group)) +
  geom_point(alpha = 0.6, size = 1.2) +
  geom_smooth(se = FALSE, method = "loess", span = span) +
  labs(
    x = "Epoch",
    y = "Validation Accuracy",
    color = "Train group",
    title = "Learning Curve (Validation Accuracy)"
  ) +
  theme_minimal()

ggsave(out_png, plot = p, width = 8, height = 5, dpi = 200)
cat("Saved plot:", out_png, "\n")
cat("Saved table:", out_csv, "\n")
