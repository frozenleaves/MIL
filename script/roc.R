#!/usr/bin/env Rscript

# 从 calibration_long.csv（sample_id,label,y_true,y_prob）绘制“micro-average ROC”论文风格图：
# - 仅一条 ROC 曲线（micro-average：把每个(样本,类别)当成一次二分类观测）
# - 随机分类器对角线
# - 曲线下面积阴影
# - AUROC 标注
#
# 兼容 R 3.6（不依赖 dplyr/readr/pROC 等包）

suppressPackageStartupMessages({
  library(ggplot2)
})

args <- commandArgs(trailingOnly = TRUE)

in_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script"
out_dir <- "C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/roc"
if (!dir.exists(out_dir)) {
  dir.create(out_dir, recursive = TRUE)
}

csv_files <- list.files(in_dir, pattern = "^calibration.*\\.csv$", full.names = TRUE)
if (length(csv_files) == 0) {
  stop("未找到 csv 文件：", in_dir)
}

for (in_path in csv_files) {
  out_path <- file.path(out_dir, paste0(tools::file_path_sans_ext(basename(in_path)), ".png"))

  df <- read.csv(in_path, stringsAsFactors = FALSE)
  if (!all(c("y_true", "y_prob") %in% colnames(df))) {
    warning("输入 CSV 缺少列 y_true / y_prob，跳过：", in_path)
    next
  }

  y <- as.integer(df$y_true)
  p <- as.numeric(df$y_prob)
  ok <- !is.na(y) & !is.na(p)
  y <- y[ok]
  p <- p[ok]
  p <- pmin(pmax(p, 0), 1)

  pos <- sum(y == 1)
  neg <- sum(y == 0)
  if (pos == 0 || neg == 0) {
    warning("y_true 需要同时包含 0 和 1，跳过：", in_path)
    next
  }

  # ROC 点：按概率从高到低扫阈值
  ord <- order(p, decreasing = TRUE)
  y_s <- y[ord]
  p_s <- p[ord]

  tp <- cumsum(y_s == 1)
  fp <- cumsum(y_s == 0)
  tpr <- tp / pos
  fpr <- fp / neg

  # 补齐端点
  fpr <- c(0, fpr, 1)
  tpr <- c(0, tpr, 1)

  # AUC（梯形法）
  auc <- sum((fpr[-1] - fpr[-length(fpr)]) * (tpr[-1] + tpr[-length(tpr)]) / 2)

  roc_df <- data.frame(fpr = fpr, tpr = tpr)

  # 论文风格配色（蓝线 + 浅蓝填充）
  col_line <- "#2F6DB3"
  col_fill <- "#CFE3F6"
  col_diag <- "#7A7A7A"

  p_roc <- ggplot(roc_df, aes(x = fpr, y = tpr)) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = col_diag, size = 0.9) +
    geom_ribbon(aes(ymin = 0, ymax = tpr), fill = col_fill, alpha = 0.75) +
    geom_line(color = col_line, size = 0.7) +
    coord_equal(xlim = c(0, 1), ylim = c(0, 1), expand = FALSE) +
    labs(
      x = "False Positive Rate (FPR)",
      y = "True Positive Rate (Sensitivity)"
    ) +
    theme_classic(base_size = 12) +
    theme(
      axis.title = element_text(color = "black"),
      axis.text = element_text(color = "black")
    )

  ann <- sprintf("AUROC = %.4f", auc)
  p_roc <- p_roc + annotate("label", x = 0.62, y = 0.08, label = ann,
                            size = 3.6, label.size = 0.3, fill = "white")

  # (a) 面板标记（方便你后期拼图）
  p_roc <- p_roc + annotate("text", x = 0.5, y = 1.04, label = "(a)", size = 5)

  ggsave(out_path, p_roc, width = 6.2, height = 5.2, dpi = 600, bg = "white")
  cat(sprintf("Wrote: %s\n", out_path))
}

