#!/usr/bin/env Rscript

# 基于同目录下 7 个 CSV 绘制雷达图（不同模态组合）
# 输入 CSV 列：sample_id,label,y_true,y_prob

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
})

in_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script"
out_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/radar"

if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

csv_files <- list.files(in_dir, pattern = "^calibration.*\\.csv$", full.names = TRUE)
csv_files <- sort(csv_files)

if (length(csv_files) == 0) {
  stop("未找到 csv 文件：", in_dir)
}

calc_auc <- function(y, p) {
  y <- as.integer(y)
  p <- as.numeric(p)
  pos <- sum(y == 1)
  neg <- sum(y == 0)
  if (pos == 0 || neg == 0) return(NA_real_)
  r <- rank(p, ties.method = "average")
  (sum(r[y == 1]) - pos * (pos + 1) / 2) / (pos * neg)
}

calc_metrics <- function(df) {
  if (!all(c("y_true", "y_prob") %in% colnames(df))) {
    stop("输入 CSV 缺少列 y_true / y_prob")
  }
  y <- as.integer(df$y_true)
  p <- as.numeric(df$y_prob)
  ok <- !is.na(y) & !is.na(p)
  y <- y[ok]
  p <- p[ok]
  p <- pmin(pmax(p, 0), 1)

  pred <- ifelse(p >= 0.5, 1L, 0L)
  tp <- sum(pred == 1 & y == 1)
  tn <- sum(pred == 0 & y == 0)
  fp <- sum(pred == 1 & y == 0)
  fn <- sum(pred == 0 & y == 1)
  n <- tp + tn + fp + fn

  sens <- ifelse((tp + fn) == 0, NA_real_, tp / (tp + fn))
  spec <- ifelse((tn + fp) == 0, NA_real_, tn / (tn + fp))
  prec <- ifelse((tp + fp) == 0, NA_real_, tp / (tp + fp))
  recall <- sens
  f1 <- ifelse(is.na(prec) | is.na(recall) | (prec + recall) == 0, NA_real_,
               2 * prec * recall / (prec + recall))
  bal_acc <- ifelse(is.na(sens) | is.na(spec), NA_real_, (sens + spec) / 2)

  po <- ifelse(n == 0, NA_real_, (tp + tn) / n)
  pe <- ifelse(n == 0, NA_real_,
               ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (n * n))
  kappa <- ifelse(is.na(po) | is.na(pe) | (1 - pe) == 0, NA_real_, (po - pe) / (1 - pe))

  auc <- calc_auc(y, p)

  tibble(
    metric = c("Balanced Accuracy", "Sensitivity", "Specificity", "AUC", "Kappa", "F1", "Recall"),
    value = c(bal_acc, sens, spec, auc, kappa, f1, recall)
  )
}

metric_levels <- c("Balanced Accuracy", "Sensitivity", "Specificity", "AUC", "Kappa", "F1", "Recall")

metrics_df <- lapply(csv_files, function(path) {
  df <- read_csv(path, show_col_types = FALSE)
  m <- calc_metrics(df)
  base_name <- tools::file_path_sans_ext(basename(path))
  m$modality <- sub("^calibration", "radar", base_name)
  m
}) %>%
  bind_rows() %>%
  mutate(
    metric = factor(metric, levels = metric_levels),
    value = pmin(pmax(value, 0), 1)
  )

modality_levels <- unique(metrics_df$modality)
palette_vals <- setNames(scales::hue_pal()(length(modality_levels)), modality_levels)

metric_info <- tibble(
  metric = metric_levels,
  angle = pi / 2 - seq(0, 2 * pi, length.out = length(metric_levels) + 1)[1:length(metric_levels)]
)

plot_df <- metrics_df %>%
  left_join(metric_info, by = "metric") %>%
  mutate(
    x = value * cos(angle),
    y = value * sin(angle)
  )

# 闭合多边形
plot_df_closed <- do.call(rbind, lapply(split(plot_df, plot_df$modality), function(d) {
  d <- d[match(metric_levels, d$metric), ]
  rbind(d, d[1, ])
}))

grid_levels <- seq(0.2, 1, 0.2)
circle_df <- do.call(rbind, lapply(grid_levels, function(r) {
  t <- seq(0, 2 * pi, length.out = 200)
  data.frame(x = r * cos(t), y = r * sin(t), r = r)
}))

axis_df <- data.frame(
  x = 0,
  y = 0,
  xend = cos(metric_info$angle),
  yend = sin(metric_info$angle),
  metric = metric_info$metric
)

label_df <- metric_info %>%
  mutate(
    x = 1.08 * cos(angle),
    y = 1.08 * sin(angle)
  )

plot_radar <- function(plot_df, plot_df_closed, plot_df_points, out_path) {
  p <- ggplot() +
  geom_path(data = circle_df, aes(x = x, y = y, group = r), color = "grey85") +
  geom_segment(data = axis_df, aes(x = x, y = y, xend = xend, yend = yend), color = "grey80") +
  geom_polygon(data = plot_df_closed,
               aes(x = x, y = y, group = modality, color = modality, fill = modality),
               alpha = 0.15, linewidth = 0.8, show.legend = FALSE) +
    geom_path(data = plot_df_closed, aes(x = x, y = y, group = modality, color = modality),
              linewidth = 0.9) +
    geom_point(data = plot_df_points, aes(x = x, y = y, color = modality), size = 1.6,
               show.legend = FALSE) +
  geom_text(data = label_df, aes(x = x, y = y, label = metric), size = 3.5) +
  coord_equal() +
  xlim(-1.12, 1.12) +
  ylim(-1.12, 1.12) +
    scale_color_manual(values = palette_vals, drop = FALSE) +
    scale_fill_manual(values = palette_vals, drop = FALSE) +
  theme_void(base_size = 12) +
  theme(
    legend.position = "bottom",
    legend.direction = "horizontal",
    legend.justification = "center",
    legend.box = "horizontal",
    legend.key.width = grid::unit(1.0, "lines"),
    legend.key.height = grid::unit(0.8, "lines"),
    legend.text = element_text(size = 8),
    legend.title = element_blank(),
    legend.spacing.x = grid::unit(0.4, "lines"),
    plot.margin = margin(14, 14, 20, 14),
    legend.key = element_blank(),
    legend.background = element_blank()
  ) +
  guides(
    color = guide_legend(nrow = 2, byrow = TRUE, override.aes = list(linetype = 1, shape = NA)),
    fill = "none"
  )

  ggsave(out_path, p, width = 6.8, height = 6.2, dpi = 600, bg = "white")
  cat(sprintf("Wrote: %s\n", out_path))
}

# 总图
plot_radar(plot_df, plot_df_closed, plot_df, file.path(out_dir, "radar_modalities-all.png"))

# 单图
for (mod in modality_levels) {
  df_m <- plot_df %>% filter(modality == mod)
  df_m_closed <- plot_df_closed %>% filter(modality == mod)
  out_path <- file.path(out_dir, paste0(mod, ".png"))
  plot_radar(df_m, df_m_closed, df_m, out_path)
}

