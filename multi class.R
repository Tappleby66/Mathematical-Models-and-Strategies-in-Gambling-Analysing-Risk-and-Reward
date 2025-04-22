library(ggplot2)
library(dplyr)
library(gridExtra)

home_form <- c(2,1,-1,3,2,0,1,0,-2,-1)
result <- c("Win", "Draw", "Loss", "Win", "Loss", "Draw", "Win", "Loss", "Draw", "Win")
df <- data.frame(home_form, result)
print(df)
# Plot 1 vs All for "Win"
df$win_vs_all <- ifelse(df$result == "Win", "Win", "Not Win")

p0 <- ggplot(df, aes(x = home_form , y = result)) +
  geom_point(aes(colour = result), size = 4) + 
  labs(title = "Results vs Form", x = "Home Form" , y="")+
  theme_minimal()
p1 <- ggplot(df, aes(x = home_form, y = win_vs_all)) +
  geom_point(aes(color = win_vs_all), size = 4) +
  labs(title = "Win vs Not Win", x = "Home Form", y = "") +
  theme_minimal() +
  scale_color_manual(values = c("Win" = "green", "Not Win" = "gray"))

# Plot 1 vs All for "Draw"
df$draw_vs_all <- ifelse(df$result == "Draw", "Draw", "Not Draw")

p2 <- ggplot(df, aes(x = home_form, y = draw_vs_all)) +
  geom_point(aes(color = draw_vs_all), size = 4) +
  labs(title = "Draw vs Not Draw", x = "Home Form", y = "") +
  theme_minimal() +
  scale_color_manual(values = c("Draw" = "orange", "Not Draw" = "gray"))

# Plot 1 vs All for "Loss"
df$loss_vs_all <- ifelse(df$result == "Loss", "Loss", "Not Loss")

p3 <- ggplot(df, aes(x = home_form, y = loss_vs_all)) +
  geom_point(aes(color = loss_vs_all), size = 4) +
  labs(title = "Loss vs Not Loss", x = "Home Form", y = "") +
  theme_minimal() +
  scale_color_manual(values = c("Loss" = "red", "Not Loss" = "gray"))

# Save plots
ggsave("result.pdf",p0,width = 6 , height = 4)
ggsave("win_vs_all.pdf", p1, width = 6, height = 4)
ggsave("draw_vs_all.pdf", p2, width = 6, height = 4)
ggsave("loss_vs_all.pdf", p3, width = 6, height = 4)


pdf("combined_plot.pdf", width = 10, height = 8)
grid.arrange(p0, p1, p2, p3, ncol = 2)
dev.off()
