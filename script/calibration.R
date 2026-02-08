#!/usr/bin/env Rscript

# 论文风格校准曲线（micro-average）：
# - 平滑校准曲线（loess）
# - 80%/95% 置信带（bootstrap）
# - Hosmer-Lemeshow 检验 P 值
#
# 输入：Python 导出的长表 calibration_long.csv
#   sample_id,label,y_true,y_prob
#
# 输出：PNG（600 dpi）

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ResourceSelection)
})

args <- commandArgs(trailingOnly = TRUE)
B <- ifelse(length(args) >= 1, as.integer(args[[1]]), 1000)

in_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script"
out_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/Calibration_Curve"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

csv_files <- list.files(in_dir, pattern = "^calibration.*\\.csv$", full.names = TRUE)
if (length(csv_files) == 0) {
  stop("未找到 csv 文件：", in_dir)
}

for (in_path in csv_files) {
  out_path <- file.path(out_dir, paste0(tools::file_path_sans_ext(basename(in_path)), ".png"))

  df <- read_csv(in_path, show_col_types = FALSE) %>%
    mutate(
      y_true = as.integer(y_true),
      y_prob = pmin(pmax(as.numeric(y_prob), 0), 1)
    ) %>%
    filter(!is.na(y_true), !is.na(y_prob))

  if (nrow(df) < 100) {
    warning("数据量过少，跳过：", in_path)
    next
  }

  # n：唯一 sample 数（更像论文里写的 n）
  n_unique <- df %>% distinct(sample_id) %>% nrow()

  # AUC（micro-average）：用秩统计计算，避免依赖额外包
  calc_auc <- function(y, p) {
    y <- as.integer(y)
    p <- as.numeric(p)
    pos <- sum(y == 1)
    neg <- sum(y == 0)
    if (pos == 0 || neg == 0) return(NA_real_)
    r <- rank(p, ties.method = "average")
    (sum(r[y == 1]) - pos * (pos + 1) / 2) / (pos * neg)
  }
  auc_val <- calc_auc(df$y_true, df$y_prob)

  # Hosmer-Lemeshow（按概率分组，默认10组）
  hl <- tryCatch({
    ResourceSelection::hoslem.test(df$y_true, df$y_prob, g = 10)
  }, error = function(e) NULL)

  hl_p <- ifelse(is.null(hl), NA_real_, as.numeric(hl$p.value))
  n_obs <- nrow(df)

  # 画图用 grid
  grid <- tibble(p = seq(0, 1, length.out = 200))

  fit_loess <- function(d) {
    # degree=1 更接近文献那种“线性/单调”外观
    loess(y_true ~ y_prob, data = d, degree = 1, span = 0.75, control = loess.control(surface = "direct"))
  }

  pred_on_grid <- function(model) {
    predict(model, newdata = data.frame(y_prob = grid$p))
  }

  # bootstrap 置信带
  set.seed(42)
  boot_mat <- matrix(NA_real_, nrow = nrow(grid), ncol = B)
  for (b in seq_len(B)) {
    idx <- sample.int(nrow(df), size = nrow(df), replace = TRUE)
    d_b <- df[idx, , drop = FALSE]
    m <- tryCatch(fit_loess(d_b), error = function(e) NULL)
    if (!is.null(m)) {
      yhat <- pred_on_grid(m)
      boot_mat[, b] <- pmin(pmax(yhat, 0), 1)
    }
  }

  q <- function(p) apply(boot_mat, 1, quantile, probs = p, na.rm = TRUE, names = FALSE)
  band <- grid %>%
    mutate(
      lo80 = q(0.10),
      hi80 = q(0.90),
      lo95 = q(0.025),
      hi95 = q(0.975)
    )

  # 主拟合
  model_full <- fit_loess(df)
  curve <- grid %>%
    mutate(y = pmin(pmax(pred_on_grid(model_full), 0), 1))

  # 颜色尽量贴近 NPJ 常见蓝色系
  col_line <- "#2F6DB3"
  col_95 <- "#C7DBF6"
  col_80 <- "#7FA8F0"

  p <- ggplot() +
    # 先画 95% 再画 80%，确保 80% 在上层更显眼
    geom_ribbon(data = band, aes(x = p, ymin = lo95, ymax = hi95, fill = "95%"), alpha = 0.85) +
    geom_ribbon(data = band, aes(x = p, ymin = lo80, ymax = hi80, fill = "80%"), alpha = 0.75) +
    geom_line(data = curve, aes(x = p, y = y), color = col_line, linewidth = 1.2) +
    geom_abline(slope = 1, intercept = 0, linetype = "dotted", color = "black", linewidth = 0.9) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    scale_fill_manual(
      name = "Confidence level",
      values = c("95%" = col_95, "80%" = col_80)
    ) +
    labs(
      x = "Predicted Probability",
      y = "Actual Probability"
    ) +
    theme_classic(base_size = 12) +
    theme(
      legend.position = c(0.86, 0.14),
      legend.background = element_rect(fill = "white", color = "grey80"),
      legend.title = element_text(size = 10),
      legend.text = element_text(size = 9),
      axis.title = element_text(color = "black"),
      axis.text = element_text(color = "black")
    )

  ann <- sprintf(
    "Degree Fit     %d\nn              %d\nAUC            %s\nH-L p          %s",
    1,
    n_unique,
    ifelse(is.na(auc_val), "NA", formatC(auc_val, format = "f", digits = 3)),
    ifelse(is.na(hl_p), "NA", formatC(hl_p, format = "f", digits = 3))
  )

  p <- p + annotate("label", x = 0.02, y = 0.98, hjust = 0, vjust = 1,
                    label = ann, size = 3.4, label.size = 0.3, fill = "white")

  # (b) 面板标记（方便你后期拼图）
  p <- p + annotate("text", x = 0.5, y = 1.04, label = "(b)", size = 5)

  ggsave(out_path, p, width = 6.2, height = 5.2, dpi = 600, bg = "white")
  cat(sprintf("Wrote: %s\n", out_path))
}

