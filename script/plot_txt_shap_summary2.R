library(readr)
library(dplyr)
library(ggplot2)
# 可选：更像 beeswarm 的排布
# install.packages("ggbeeswarm")
library(ggbeeswarm)
library(forcats)

df <- read_csv("C:/Users/frozen/Desktop/20260113实验需求/MIL/script/txt_ig_shap_long.csv", show_col_types = FALSE)

TOP_N <- 40

plot_df <- df %>%
  group_by(token_decoded) %>%
  summarise(mean_abs = mean(abs(shap_value)), .groups="drop") %>%
  slice_max(mean_abs, n = TOP_N, with_ties = FALSE) %>%
  inner_join(df, by = c("token_decoded")) %>%
  mutate(
    token = fct_reorder(token_decoded, mean_abs),
    feature_value = pmin(token_count, 5) # 颜色用出现次数（压缩一下）
  )

p <- ggplot(plot_df, aes(x = shap_value, y = token, color = feature_value)) +
  geom_violin(fill = "grey92", color = NA, scale = "width", trim = TRUE) +
  geom_quasirandom(groupOnX = FALSE, alpha = 0.65, size = 1.4, width = 0.25) +
  geom_vline(xintercept = 0, color = "grey40", linewidth = 0.4) +
  scale_color_gradient(low = "#2C7BB6", high = "#D7191C", name = "Feature value") +
  labs(x = "Attribution (impact on model output)", y = NULL) +
  theme_minimal(base_size = 12) +
  theme(panel.grid.minor = element_blank())

ggsave("C:/Users/frozen/Desktop/20260113实验需求/MIL/script/figures/Txt_Shap_Summary/txt_shap_summary_like2.png", p, width = 10, height = 6, dpi = 200)